"""
Real-time answer revision (live-conversational-intelligence R31).

When a follow-up reveals the prior question was misinterpreted ("I meant the
Redis cluster", "no, I was asking about X"), `detect_reinterpretation` returns
the prior answer's `qid` so the live loop can emit an additive `revision` event
targeting that answer (updated in place, not a duplicate) rather than treating
the turn as a brand-new question. Deterministic + fail-open.
"""
from __future__ import annotations

import re

from app.core import lexicons

_REINTERPRET_CUES = lexicons.LIVE_REVISE_REINTERPRET_CUES


def detect_reinterpretation(turn: str, world_model) -> str | None:
    """Return the active answer's `qid` to revise when `turn` reinterprets the
    prior question; else None. Never raises."""
    try:
        t = (turn or "").strip().lower()
        if not t:
            return None
        if not any(c in t for c in _REINTERPRET_CUES):
            return None
        qid = getattr(world_model, "active_qid", "") or ""
        return qid or None
    except Exception:  # noqa: BLE001
        return None


def revised_question(turn: str, world_model) -> str:
    """Build the corrected question from the reinterpretation turn + the prior
    active question. Best-effort; falls back to the turn itself."""
    try:
        prior = (getattr(world_model, "active_question", "") or "").strip()
        t = (turn or "").strip()
        # Extract the corrected subject after the cue, if present.
        m = re.search(r"(?:i meant|i was (?:asking about|referring to|talking about)|"
                      r"i mean)\s+(.+)", t, re.IGNORECASE)
        subject = m.group(1).strip().rstrip(".?!") if m else ""
        if prior and subject:
            return f"{prior} (clarified: about {subject})"
        return t or prior
    except Exception:  # noqa: BLE001
        return turn or ""
