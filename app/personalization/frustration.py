"""Frustration detection (personalization-and-governance R3).

Raises a 0..1 frustration signal on repeated rephrasing / corrections / negative
feedback within a window, decays it back toward baseline when interactions
normalize, and exposes a bias hint (prefer concise/direct + fewer clarifications
while elevated) that COMPOSES with the existing fatigue/trust model. Never
suppresses a blocking safety/destructive confirmation (R3.4, Property 3). Pure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_RISE_CORRECTION = 0.3
_RISE_NEGATIVE = 0.4
_RISE_REPHRASE = 0.2
_DECAY = 0.25
_ELEVATED = 0.5

_CORRECTION_RE = re.compile(
    r"\b(no|not|wrong|that's not|thats not|actually|i meant|i said|"
    r"that isn'?t|incorrect|still (?:wrong|not))\b", re.I)
_NEGATIVE_RE = re.compile(
    r"\b(useless|terrible|awful|stupid|doesn'?t work|not working|broken|"
    r"frustrat|annoying|come on|ugh|seriously)\b", re.I)


@dataclass
class FrustrationState:
    value: float = 0.0

    @property
    def elevated(self) -> bool:
        return self.value >= _ELEVATED


def _similar(a: str, b: str) -> bool:
    """Cheap rephrase detector: high token overlap between consecutive turns."""
    ta = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) > 2}
    if not ta or not tb:
        return False
    return len(ta & tb) / float(len(ta | tb)) >= 0.6


def update_frustration(state: FrustrationState, turn: str, *,
                       prev_turn: str = "", negative_feedback: bool = False
                       ) -> FrustrationState:
    """Update the frustration signal for this turn. Rises on correction /
    negative feedback / rephrase; otherwise decays. Never raises."""
    try:
        v = state.value
        bumped = False
        if negative_feedback:
            v += _RISE_NEGATIVE
            bumped = True
        if _NEGATIVE_RE.search(turn or ""):
            v += _RISE_NEGATIVE
            bumped = True
        if _CORRECTION_RE.search(turn or ""):
            v += _RISE_CORRECTION
            bumped = True
        if prev_turn and _similar(turn, prev_turn):
            v += _RISE_REPHRASE
            bumped = True
        if not bumped:
            v -= _DECAY                      # normalize → decay (R3.3)
        return FrustrationState(value=max(0.0, min(1.0, v)))
    except Exception:  # noqa: BLE001
        return state


def bias(state: FrustrationState) -> dict:
    """The adjustment to apply while frustrated: terser answers + fewer
    clarifications. Additive guidance, never a safety override (R3.2/R3.4)."""
    if not state.elevated:
        return {}
    return {
        "prefer_concise": True,
        "fewer_clarifications": True,
        "directive": ("The user appears frustrated — answer concisely and "
                      "directly, avoid extra clarifying questions, and get to "
                      "the fix first."),
    }


__all__ = ["FrustrationState", "update_frustration", "bias"]
