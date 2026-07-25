"""Structured Solve extraction + language ladder (vNext §3.2).

Replaces the free-text delimited OCR (`=== TITLE === …`) with a JSON-schema
**contract** so the language decision and the clarifier become deterministic:

    {platform, title, statement, constraints[], examples[], selected_language,
     starter_code, ui_confidence{selected_language, examples}}

The VLM runs on the vision model (via the `app.core.llm_client` boundary) and the
result is validated + repaired through the §8.7 helper (`app.core.structured.
parse_with_repair`) — a schema-valid object or None (fail-open → the caller keeps
today's delimited OCR path). Language then resolves down a strict ladder:

    explicit request  →  selected_language (conf ≥ threshold)  →  starter-code
    inference  →  session sticky  →  (caller) clarifier.

Lives in `codeintel` (code intelligence): it reaches the LLM only through the
`core` facades (existing `codeintel → core` edge) and uses `code_language`
intra-package — no new cross-package coupling.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# selected_language/starter_code are plain strings ("" = none) rather than
# nullable unions — the dependency-free validator handles this cleanly, and the
# caller treats "" as absent.
SOLVE_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "platform": {"type": "string"},
        "title": {"type": "string"},
        "statement": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                    "explanation": {"type": "string"},
                },
            },
        },
        "selected_language": {"type": "string"},
        "starter_code": {"type": "string"},
        "ui_confidence": {
            "type": "object",
            "properties": {
                "selected_language": {"type": "number"},
                "examples": {"type": "number"},
            },
        },
    },
    "required": ["title", "statement"],
}

_EXTRACT_SYSTEM = (
    "You are an exact OCR + structure extractor for a coding-interview "
    "screenshot. Read the image and return ONLY a JSON object matching the "
    "given schema. Transcribe text VERBATIM (preserve math, symbols, and the "
    "function signature). Read the editor's SELECTED-LANGUAGE chip/dropdown and "
    "put it in `selected_language` with a 0..1 `ui_confidence.selected_language` "
    "(1.0 = the chip is clearly legible, 0.0 = not visible). Copy any visible "
    "starter code into `starter_code`. Do NOT solve, summarize, translate, or "
    "invent anything — extraction only."
)
_EXTRACT_USER = (
    "Extract this coding problem into the JSON schema. Transcribe exactly; leave "
    "a field empty ('' or []) when it isn't visible."
)


@dataclass
class ExtractedProblem:
    title: str
    statement: str
    platform: str = "unknown"
    constraints: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    selected_language: str = ""
    starter_code: str = ""
    lang_confidence: float = 0.0
    examples_confidence: float = 0.0

    @classmethod
    def from_obj(cls, obj: dict) -> "ExtractedProblem | None":
        if not isinstance(obj, dict):
            return None
        title = str(obj.get("title") or "").strip()
        statement = str(obj.get("statement") or "").strip()
        if not statement:
            return None
        conf = obj.get("ui_confidence") or {}
        exs = obj.get("examples") or []
        return cls(
            title=title or "(untitled)",
            statement=statement,
            platform=str(obj.get("platform") or "unknown").strip().lower(),
            constraints=[str(c) for c in (obj.get("constraints") or []) if c],
            examples=[e for e in exs if isinstance(e, dict)],
            selected_language=str(obj.get("selected_language") or "").strip(),
            starter_code=str(obj.get("starter_code") or ""),
            lang_confidence=_as_float(conf.get("selected_language")),
            examples_confidence=_as_float(conf.get("examples")),
        )

    def to_delimited(self) -> str:
        """Render back to the `=== HEADING ===` form the reasoning step already
        consumes, so the downstream pipeline is unchanged."""
        lines = [f"=== TITLE ===\n{self.title}"]
        if self.starter_code.strip():
            lines.append(f"=== FUNCTION SIGNATURE ===\n{self.starter_code}")
        lines.append(f"=== PROBLEM STATEMENT ===\n{self.statement}")
        if self.examples:
            ex_txt = "\n\n".join(
                "Input: {i}\nOutput: {o}{e}".format(
                    i=e.get("input", ""), o=e.get("output", ""),
                    e=("\nExplanation: " + e["explanation"])
                    if e.get("explanation") else "")
                for e in self.examples)
            lines.append(f"=== EXAMPLES ===\n{ex_txt}")
        if self.constraints:
            lines.append("=== CONSTRAINTS ===\n" + "\n".join(self.constraints))
        if self.starter_code.strip():
            lines.append(f"=== STARTER CODE ===\n{self.starter_code}")
        return "\n\n".join(lines)

    def summary(self) -> str:
        """A one-line progressive-step label ("Read the problem — Two Sum, …")."""
        bits = [self.title]
        if self.selected_language:
            bits.append(self.selected_language)
        if self.examples:
            bits.append(f"{len(self.examples)} example"
                        f"{'s' if len(self.examples) != 1 else ''}")
        return "Read the problem — " + ", ".join(b for b in bits if b)


def _as_float(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


async def extract_structured(
    image_bytes: bytes, *, vision_model: str | None = None,
    extra_context: str | None = None, retries: int = 1,
) -> ExtractedProblem | None:
    """Run the VLM with the JSON-schema contract and validate/repair the result.
    Returns an :class:`ExtractedProblem` or None (fail-open). Never raises."""
    try:
        from app.core.config_loader import cfg
        from app.core.llm_client import llm
        from app.core.structured import parse_with_repair
    except Exception:  # noqa: BLE001
        return None
    model = vision_model or cfg.llm.vision_model or cfg.llm.model
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
    except Exception:  # noqa: BLE001
        return None
    import json as _json
    user = _EXTRACT_USER + (f"\n\nHint: {extra_context}" if extra_context else "")
    user += "\n\nJSON Schema:\n" + _json.dumps(SOLVE_EXTRACTION_SCHEMA,
                                               sort_keys=True)
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user, "images": [encoded]},
    ]
    for attempt in range(max(1, retries + 1)):
        try:
            text = await llm.complete(
                messages, model=model,
                options={"temperature": 0.0,
                         "num_predict": int(getattr(cfg.code_solver,
                                                    "ocr_max_tokens", 1500))})
        except Exception as exc:  # noqa: BLE001
            log.info("solve_extract: VLM call failed (attempt %d): %s",
                     attempt, exc)
            return None
        obj, errs = parse_with_repair(text or "", SOLVE_EXTRACTION_SCHEMA)
        if obj is not None and not errs:
            got = ExtractedProblem.from_obj(obj)
            if got is not None:
                return got
        if attempt < retries:
            messages = messages + [
                {"role": "assistant", "content": (text or "")[:1500]},
                {"role": "user", "content": (
                    "That was not valid against the schema. Return ONLY corrected "
                    "JSON — no prose, no fences.")},
            ]
    return None


def resolve_language(
    extracted: "ExtractedProblem | None", *,
    requested: str | None = None, sticky: str | None = None,
    threshold: float | None = None,
) -> tuple[str | None, str]:
    """The §3.2 language ladder. Returns ``(language_label, source)`` where
    ``source`` is one of ``requested|selected|starter|sticky|none``. ``None`` →
    the caller fires the clarifier (only a truly-unknown language blocks)."""
    if threshold is None:
        try:
            from app.core.config_loader import cfg
            threshold = float(getattr(cfg.code_solver,
                                      "selected_language_min_conf", 0.7))
        except Exception:  # noqa: BLE001
            threshold = 0.7

    # 1) An explicit user request always wins.
    if requested and requested.strip():
        return requested.strip(), "requested"
    if extracted is not None:
        # 2) The editor's selected-language chip, when confidently read.
        if (extracted.selected_language
                and extracted.lang_confidence >= threshold):
            return extracted.selected_language, "selected"
        # 3) Infer from the visible starter code.
        if extracted.starter_code.strip():
            try:
                from app.codeintel.code_language import (
                    detect_language, detect_language_label)
                lbl = (detect_language_label(extracted.starter_code)
                       or detect_language(extracted.starter_code))
                if lbl:
                    return lbl, "starter"
            except Exception:  # noqa: BLE001
                pass
    # 4) The conversation's sticky language.
    if sticky and sticky.strip():
        return sticky.strip(), "sticky"
    # 5) Unknown → clarifier.
    return None, "none"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.code_solver, "structured_extraction", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "SOLVE_EXTRACTION_SCHEMA", "ExtractedProblem", "extract_structured",
    "resolve_language", "enabled",
]
