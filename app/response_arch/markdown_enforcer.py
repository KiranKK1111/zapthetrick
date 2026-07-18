"""Guarantee well-formed markdown for the configured shape.

Models drift — and the free GPT-OSS-class models drift a LOT on block
structure: they emit tables and headings with **no line breaks**, e.g.

    ...for delegates ### When to use | Situation | Reason | | --- | --- | | a | b |

When that reaches the renderer, the table isn't recognized (a GFM table
needs each row on its own line) so it's treated as one paragraph and the
newlines collapse to spaces — the "flattened table" the user sees.

This module repairs that. The heavy lifters:
  * `_reflow_tables`   — reconstruct a single-line (flattened) table into
    proper newline-separated rows with a blank line around it.
  * `_deglue_headings` — put a heading that got glued mid-line onto its own
    line.

All repairs skip fenced code blocks so we never touch code that legitimately
contains `|` or `#`. The rules are conservative: a well-formed (already
multi-line) table is left untouched, and anything that doesn't clearly parse
as a table is left alone.
"""
from __future__ import annotations

import re

from .content_router import Shape


def enforce_markdown(text: str, *, shape: Shape) -> str:
    """Run the rule chain. Repairs run on prose segments only (not code)."""
    if not text:
        return text

    out_parts: list[str] = []
    for is_code, seg in _split_code_fences(text):
        if is_code:
            out_parts.append(seg)
            continue
        s = seg
        s = _normalize_bullets(s)
        s = _normalize_headings(s)
        s = _strip_heading_indent(s)
        s = _deglue_headings(s)
        s = _reflow_tables(s)
        if shape == Shape.STEPS:
            s = _ensure_ordered_list(s)
        out_parts.append(s)

    out = "".join(out_parts)
    out = _close_unbalanced_fences(out)
    return out


# ---- code-fence segmentation -------------------------------------------
_FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)


def _split_code_fences(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, segment) parts, keeping fenced blocks intact."""
    parts: list[tuple[bool, str]] = []
    last = 0
    for m in _FENCE_RE.finditer(text):
        if m.start() > last:
            parts.append((False, text[last:m.start()]))
        parts.append((True, m.group(0)))
        last = m.end()
    if last < len(text):
        parts.append((False, text[last:]))
    return parts or [(False, text)]


# ---- rules -------------------------------------------------------------
def _close_unbalanced_fences(text: str) -> str:
    """If the number of ``` markers is odd, append a closing fence."""
    if text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```\n"
    return text


_BULLET_RE = re.compile(r"^(\s*)([\*•·])\s+", re.MULTILINE)


def _normalize_bullets(text: str) -> str:
    """Normalize bullet marker to `-`."""
    return _BULLET_RE.sub(r"\1- ", text)


_HEADING_NOSPACE_RE = re.compile(r"^(#{1,6})([A-Za-z])", re.MULTILINE)


def _normalize_headings(text: str) -> str:
    """Models occasionally write `##Heading` (no space). Fix that."""
    return _HEADING_NOSPACE_RE.sub(r"\1 \2", text)


_HEADING_INDENT_RE = re.compile(r"^[ \t]+(#{1,6}[ \t])", re.MULTILINE)


def _strip_heading_indent(text: str) -> str:
    """Drop leading spaces before a heading so it parses as one."""
    return _HEADING_INDENT_RE.sub(r"\1", text)


# A heading glued after sentence punctuation / a table pipe, e.g. ".### When"
# or "| ### Security".
_DEGLUE_PUNCT_RE = re.compile(r"([.!?:;|>)\]])[ \t]*(#{1,6}[ \t]+\S)")
# A heading (## or deeper) glued straight onto a word, e.g. "delegates### When".
# Restricted to >=2 hashes so we never split "C#" / "F#".
_DEGLUE_WORD_RE = re.compile(r"(\w)[ \t]*(#{2,6}[ \t]+\S)")


def _deglue_headings(text: str) -> str:
    """Put a mid-line heading onto its own line (blank line before it)."""
    out = _DEGLUE_PUNCT_RE.sub(r"\1\n\n\2", text)
    out = _DEGLUE_WORD_RE.sub(r"\1\n\n\2", out)
    return out


# A run of pipe-delimited cells on a single line. `[^|\n]` keeps it within one
# physical line, so an already-multi-line (well-formed) table never matches as
# a whole — only a flattened one does.
_PIPE_RUN_RE = re.compile(r"\|(?:[^|\n]*\|)+")
_DASH_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _reflow_tables(text: str) -> str:
    """Reconstruct a flattened (single-line) GFM table into real rows.

    Only acts on a pipe run that contains a dash separator row; everything
    else is returned unchanged.
    """

    def repl(m: re.Match) -> str:
        span = m.group(0)
        cells = [c.strip() for c in span.split("|")]
        # Drop empty boundary tokens (leading/trailing pipes + the artifact
        # left between flattened rows). NOTE: this also drops legitimately
        # empty cells — acceptable; they're rare and the alternative is the
        # unreadable flattened blob.
        cells = [c for c in cells if c != ""]
        if not cells:
            return span

        # Find the contiguous separator block (all-dash cells).
        sep_start = next(
            (i for i, c in enumerate(cells) if _DASH_CELL_RE.match(c)), None
        )
        if sep_start is None or sep_start == 0:
            return span  # no separator, or no header before it → not a table
        sep_len = 0
        while (
            sep_start + sep_len < len(cells)
            and _DASH_CELL_RE.match(cells[sep_start + sep_len])
        ):
            sep_len += 1

        n_cols = sep_start  # header cells == column count
        header = cells[:n_cols]
        body = cells[sep_start + sep_len:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * n_cols) + " |",
        ]
        for i in range(0, len(body), n_cols):
            row = body[i:i + n_cols]
            if not row:
                continue
            row = row + [""] * (n_cols - len(row))  # pad a short final row
            lines.append("| " + " | ".join(row) + " |")

        return "\n\n" + "\n".join(lines) + "\n\n"

    return _PIPE_RUN_RE.sub(repl, text)


_UNORDERED_THEN_NUM_RE = re.compile(r"^(\s*)-\s+(\d+)[.)]\s+", re.MULTILINE)


def _ensure_ordered_list(text: str) -> str:
    """Convert `- 1.` style to `1.` for shape=steps."""
    return _UNORDERED_THEN_NUM_RE.sub(r"\1\2. ", text)


__all__ = ["enforce_markdown"]
