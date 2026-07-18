"""Stage 1 — Problem extractor.

Normalizes a raw problem statement into a [ProblemSpec]: title,
constraints, input/output spec, examples, language hint.

Two paths:
  - **Text** (the common case via `/api/solve/text`): we already have
    structured text; just parse it.
  - **Image** is handled upstream — the existing `code_solver.solve_image`
    runs OCR with `_OCR_SYSTEM_PROMPT` that emits the same labeled
    sections (TITLE / FUNCTION SIGNATURE / PROBLEM STATEMENT /
    EXAMPLES / CONSTRAINTS). This stage parses that format.
"""
from __future__ import annotations

import re

from .types import ProblemSpec


_SECTION_RE = re.compile(
    r"^\s*(?:===\s*)?(TITLE|FUNCTION\s*SIGNATURE|PROBLEM\s*STATEMENT|EXAMPLES?|CONSTRAINTS?|INPUT|OUTPUT)"
    r"(?:\s*===)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EXAMPLE_BLOCK_RE = re.compile(
    r"(?:Example\s*\d*|Input)\s*:?(.+?)(?=(?:Example\s*\d*|Input)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_LANG_HINT_RE = re.compile(
    r"\b(python|java(?:script)?|typescript|c\+\+|c#|go(?:lang)?|rust|kotlin|swift|ruby|php|scala|dart)\b",
    re.IGNORECASE,
)


def extract(raw: str) -> ProblemSpec:
    """Parse a free-text or OCR'd problem into a [ProblemSpec].

    Falls back to "the whole thing is the statement" when no section
    headers exist — better than dropping data.
    """
    if not raw or not raw.strip():
        return ProblemSpec(statement="")

    text = raw.strip()
    sections = _split_sections(text)

    return ProblemSpec(
        title=sections.get("TITLE", "").strip().split("\n")[0][:200],
        statement=(sections.get("PROBLEM STATEMENT") or text).strip(),
        constraints=_parse_bullets(sections.get("CONSTRAINTS", "")),
        input_spec=sections.get("INPUT", "").strip(),
        output_spec=sections.get("OUTPUT", "").strip(),
        examples=_parse_examples(sections.get("EXAMPLES", "")),
        language_hint=_detect_language(text),
    )


# ---- helpers ------------------------------------------------------------
def _split_sections(text: str) -> dict[str, str]:
    """Walk the text, splitting on the recognised section headers."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = _canonical_section(m.group(1))
            buf = []
        else:
            if current is None:
                # Before any header — treat as the statement until proven
                # otherwise.
                current = "PROBLEM STATEMENT"
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _canonical_section(label: str) -> str:
    norm = re.sub(r"\s+", " ", label.upper()).strip()
    if norm.startswith("FUNCTION"):
        return "FUNCTION SIGNATURE"
    if norm.startswith("PROBLEM"):
        return "PROBLEM STATEMENT"
    if norm.startswith("EXAMPLE"):
        return "EXAMPLES"
    if norm.startswith("CONSTRAINT"):
        return "CONSTRAINTS"
    return norm


def _parse_bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("•*-").strip()
        if stripped:
            out.append(stripped)
    return out


def _parse_examples(text: str) -> list[dict]:
    """Pull each `Example N:` block out as a dict.

    The OCR pipeline emits free-form examples (`Input: ...\nOutput:
    ...\nExplanation: ...`). We're tolerant about it — preserving the
    raw text so the solution generator has full context.
    """
    if not text.strip():
        return []
    out: list[dict] = []
    for block in re.split(r"\n\s*Example\s*\d+\s*[:\-]?\s*", "\n" + text, flags=re.IGNORECASE):
        if not block.strip():
            continue
        body = block.strip()
        ex = {"raw": body}
        m_in = re.search(r"Input\s*:\s*(.+?)(?=\n\s*(?:Output|Explanation)\s*:|\Z)", body, re.DOTALL | re.IGNORECASE)
        m_out = re.search(r"Output\s*:\s*(.+?)(?=\n\s*Explanation\s*:|\Z)", body, re.DOTALL | re.IGNORECASE)
        m_ex = re.search(r"Explanation\s*:\s*(.+)", body, re.DOTALL | re.IGNORECASE)
        if m_in:
            ex["input"] = m_in.group(1).strip()
        if m_out:
            ex["output"] = m_out.group(1).strip()
        if m_ex:
            ex["note"] = m_ex.group(1).strip()
        out.append(ex)
    return out


def _detect_language(text: str) -> str | None:
    m = _LANG_HINT_RE.search(text)
    if not m:
        return None
    return m.group(1).lower().replace("golang", "go").replace("javascript", "javascript")
