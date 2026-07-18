"""
Interviewer-satisfaction detection (live-conversational-intelligence R13).

Classifies short interviewer reactions as closing the current thread ("good",
"makes sense", "correct") or keeping it open ("not quite", "think deeper").
Deterministic + fail-open: returns None when neither applies (it is then handled
as normal speech).
"""
from __future__ import annotations

from app.core import lexicons

CLOSED = "closed"
OPEN = "open"

_CLOSED_CUES = lexicons.LIVE_SATISFACTION_CLOSED_CUES
_OPEN_CUES = lexicons.LIVE_SATISFACTION_OPEN_CUES


def classify_feedback(text: str) -> str | None:
    """Return CLOSED / OPEN / None for an interviewer reaction. Never raises."""
    try:
        t = (text or "").strip().lower()
        if not t:
            return None
        # Open (dissatisfaction) takes precedence — keep the thread alive.
        if any(c in t for c in _OPEN_CUES):
            return OPEN
        # Closing cues only count for short reactions (not long new questions).
        if len(t.split()) <= 6 and any(c in t for c in _CLOSED_CUES):
            return CLOSED
        return None
    except Exception:  # noqa: BLE001
        return None
