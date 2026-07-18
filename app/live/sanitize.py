"""
Transcript sanitization / prompt-injection guard
(live-conversational-intelligence R21).

The transcript is UNTRUSTED input. If the interviewer (or background audio / a
read-aloud snippet) contains instruction-like content ("ignore previous
instructions", "you are now …", "system prompt: …"), it must not be able to
override the answer/system prompt. `sanitize` neutralizes those spans while
keeping the genuine question intact. Deterministic + fail-open: the existing
system-prompt precedence is retained; on error the text passes through.
"""
from __future__ import annotations

import re

_FILTERED = "[filtered]"

# Instruction-like spans to neutralize (the matched clause up to sentence end).
_INJECTION = re.compile(
    r"(?i)\b("
    r"ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions?|prompts?)"
    r"|disregard (?:all |the )?(?:previous|prior|above)[^.?!]*"
    r"|forget (?:everything|all|the above|previous instructions)[^.?!]*"
    r"|you are now\b[^.?!]*"
    r"|act as\b[^.?!]*"
    r"|pretend (?:to be|you are)\b[^.?!]*"
    r"|system prompt\s*:?[^.?!]*"
    r"|new instructions?\s*:?[^.?!]*"
    r"|override (?:your |the )?(?:instructions?|system)[^.?!]*"
    r")"
)


def has_injection(text: str) -> bool:
    """True when an instruction-like span is present. Never raises."""
    try:
        return bool(_INJECTION.search(text or ""))
    except Exception:  # noqa: BLE001
        return False


def sanitize(text: str) -> str:
    """Neutralize instruction-like spans, keeping the genuine question. Never
    raises — returns the input unchanged on error."""
    if not text or not text.strip():
        return text or ""
    try:
        cleaned = _INJECTION.sub(_FILTERED, text)
        # Collapse any double spaces introduced by substitution.
        return re.sub(r"\s{2,}", " ", cleaned).strip()
    except Exception:  # noqa: BLE001
        return text
