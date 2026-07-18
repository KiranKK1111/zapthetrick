"""
Candidate delivery coaching (live-conversational-intelligence R38).

Optional, non-intrusive feedback on the candidate's OWN delivery — filler words,
answer length, missing concrete examples, and a suggested tone/duration. Additive
and never blocks/replaces the answer stream. Deterministic + fail-open: with the
flag off (or no candidate speech) nothing is surfaced.
"""
from __future__ import annotations

import re

from app.core import lexicons

_FILLERS = lexicons.LIVE_COACH_FILLERS
_EXAMPLE_CUES = lexicons.LIVE_COACH_EXAMPLE_CUES


def coach(candidate_text: str) -> list[str]:
    """Return up to a few gentle delivery tips for the candidate's utterance.
    Never raises → []."""
    try:
        t = (candidate_text or "").strip()
        if not t:
            return []
        # Strip punctuation before counting: real STT transcripts punctuate
        # ("um, so like,"), which used to defeat the space-delimited filler
        # matching entirely — filler-heavy answers got zero tips.
        import re as _re
        low = " " + _re.sub(r"[^\w\s']", " ", t.lower()) + " "
        words = t.split()
        tips: list[str] = []

        fillers = sum(low.count(" " + f + " ") for f in _FILLERS)
        if fillers >= 3:
            tips.append(f"Watch filler words ({fillers}) — pause instead.")
        if len(words) > 220:
            tips.append("Answer is long — tighten to the key points.")
        elif len(words) < 12:
            tips.append("Add a bit more detail or an example.")
        if not any(c in low for c in _EXAMPLE_CUES) and len(words) >= 20:
            tips.append("Ground it with a concrete example from your experience.")
        return tips[:3]
    except Exception:  # noqa: BLE001
        return []
