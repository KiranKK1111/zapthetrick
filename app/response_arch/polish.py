"""Tiny textual polish — runs after the markdown enforcer."""
from __future__ import annotations

import re


_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_FILLER_OPENERS_RE = re.compile(
    r"^(Sure[!,.]*|Of course[!,.]*|Certainly[!,.]*|Here[' ]?s[^\n]*\n)",
    re.IGNORECASE,
)


def polish(text: str) -> str:
    if not text:
        return text
    out = text
    out = _FILLER_OPENERS_RE.sub("", out)
    out = _TRAILING_WS_RE.sub("", out)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip() + "\n"


__all__ = ["polish"]
