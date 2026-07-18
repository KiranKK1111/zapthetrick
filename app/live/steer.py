"""Conversation Steering (roadmap Phase 2 #32).

On an OPEN prompt ("tell me about yourself", "walk me through your background"),
proactively steer the answer toward the candidate's strongest evidence instead
of a flat chronological recap. Emits an additive answer directive the responder
folds in. Deterministic + fail-open.
"""
from __future__ import annotations

# Cues that mark a broad, open-ended prompt where steering adds the most value.
_OPEN_CUES = (
    "tell me about yourself",
    "walk me through",
    "tell me about your background",
    "introduce yourself",
    "describe your experience",
    "tell me about your journey",
    "give me an overview",
    "what should i know about you",
)


def is_open_prompt(question: str) -> bool:
    try:
        q = (question or "").lower()
        return any(cue in q for cue in _OPEN_CUES)
    except Exception:  # noqa: BLE001
        return False


def steering_directive(question: str, strengths: list[str] | None = None) -> str | None:
    """Return a steering directive when the prompt is open; None otherwise.
    Names specific strengths when supplied, else emits a generic steer (the
    responder still has the profile in its own context)."""
    try:
        if not is_open_prompt(question):
            return None
        picks = [s for s in (strengths or []) if s and s.strip()][:2]
        if picks:
            lead = picks[0].strip()
            bridge = f" then bridge to {picks[1].strip()}" if len(picks) > 1 else ""
            focus = f"lead with {lead}{bridge}"
        else:
            focus = "lead with the strongest, most role-relevant experience"
        return (
            f"This is an open prompt — steer toward the candidate's strongest "
            f"evidence: {focus}, keep it a tight narrative, and end on impact, "
            f"not a full chronological recap."
        )
    except Exception:  # noqa: BLE001
        return None


__all__ = ["is_open_prompt", "steering_directive"]
