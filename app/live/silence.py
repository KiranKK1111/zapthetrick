"""
Silence taxonomy + adaptive-timeout advisory (roadmap Phase 2 #5 / 2A-5).

The endpointer already decides WHEN a turn is over (app/live/hypothesis.py).
This module classifies WHY the speaker went silent — so the pipeline can react
differently to a *thinking* pause (interviewer composing the rest of the
question), a *hesitation* (disfluency / trailing filler), and a genuine *done*
end-of-turn. Purely advisory: it never gates whether we answer (that stays with
the endpointer) — it only surfaces a `meta.silence_type` label and, at most, a
soft directive nudging the answer's opening. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

THINKING = "thinking"       # incomplete tail + a real pause → composing more
HESITATION = "hesitation"   # trailing filler / disfluency → unsure phrasing
DONE = "done"               # closed thought → genuine end-of-turn
UNKNOWN = "unknown"

# Trailing tokens that read as a disfluency the speaker paused on.
_FILLERS = ("um", "uh", "erm", "hmm", "ah", "er", "like", "so", "well", "yeah")


@dataclass
class SilenceSignal:
    label: str = UNKNOWN
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"type": self.label, "confidence": round(self.confidence, 3)}


def _trailing_filler(text: str) -> bool:
    try:
        words = [w for w in (text or "").lower().replace("?", " ").replace(".", " ").split() if w]
        return bool(words) and words[-1].strip(",;:-") in _FILLERS
    except Exception:  # noqa: BLE001
        return False


def classify(
    text: str,
    *,
    completeness: str | None = None,
    gap_ms: float | None = None,
) -> SilenceSignal:
    """Classify the silence following `text`. `completeness` (from
    hypothesis.completeness) short-circuits the lexical work when supplied.
    Never raises → UNKNOWN."""
    try:
        if completeness is None:
            from app.live.hypothesis import completeness as _c
            completeness = _c(text)
        # A trailing filler is a hesitation regardless of grammatical closure.
        if _trailing_filler(text):
            return SilenceSignal(label=HESITATION, confidence=0.8)
        if completeness == "incomplete":
            # Longer measured pause on an incomplete tail → clearly thinking.
            conf = 0.85 if (gap_ms is not None and gap_ms >= 900) else 0.7
            return SilenceSignal(label=THINKING, confidence=conf)
        if completeness == "complete":
            return SilenceSignal(label=DONE, confidence=0.8)
        # neutral tail: a long gap still reads as thinking, else unknown.
        if gap_ms is not None and gap_ms >= 1200:
            return SilenceSignal(label=THINKING, confidence=0.55)
        return SilenceSignal(label=UNKNOWN, confidence=0.0)
    except Exception:  # noqa: BLE001
        return SilenceSignal(label=UNKNOWN, confidence=0.0)


def directive(signal: SilenceSignal) -> str:
    """A soft, advisory opening nudge. '' for done/unknown or low confidence."""
    try:
        if signal.confidence < 0.6:
            return ""
        if signal.label == THINKING:
            return ("The speaker paused mid-thought — answer the question as "
                    "asked; do not invent the missing part.")
        if signal.label == HESITATION:
            return ("The speaker sounded unsure of the phrasing — open by "
                    "restating the question crisply, then answer.")
        return ""
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["SilenceSignal", "classify", "directive",
           "THINKING", "HESITATION", "DONE", "UNKNOWN"]
