"""
Interview_Mode detection & switching (live-conversational-intelligence R42).

Composes with phase detection (`app/live/phase.py`) to pick a higher-level
operating MODE that biases answer strategy: a behavioral/HR phase → STAR-story
mode, a system-design phase → structured-design mode, a coding phase →
think-aloud mode, an HR phase → negotiation-aware mode. Deterministic +
fail-open. Surfaced as additive `meta.mode`; never a second LLM call.
"""
from __future__ import annotations

from app.live import phase as _phase

# Mode identifiers.
GENERAL = "general"
STAR_STORY = "star_story"
STRUCTURED_DESIGN = "structured_design"
THINK_ALOUD = "think_aloud"
RESUME_DEEP_DIVE = "resume_deep_dive"
NEGOTIATION = "negotiation"
WRAP_UP = "wrap_up"

MODES = {GENERAL, STAR_STORY, STRUCTURED_DESIGN, THINK_ALOUD, RESUME_DEEP_DIVE,
         NEGOTIATION, WRAP_UP}

# phase → mode mapping.
_PHASE_MODE = {
    _phase.BEHAVIORAL: STAR_STORY,
    _phase.SYSTEM_DESIGN: STRUCTURED_DESIGN,
    _phase.CODING: THINK_ALOUD,
    _phase.RESUME_DISCUSSION: RESUME_DEEP_DIVE,
    _phase.HR: NEGOTIATION,
    _phase.CLOSING: WRAP_UP,
}

# Mode → answer-shaping directive.
_DIRECTIVE = {
    STAR_STORY: "Answer as a STAR story: Situation, Task, Action, Result — concrete and concise.",
    STRUCTURED_DESIGN: ("Answer as a structured design: clarify requirements, sketch components, "
                        "discuss trade-offs and scale."),
    THINK_ALOUD: "Think aloud: state the approach, complexity, and edge cases before the final solution.",
    RESUME_DEEP_DIVE: "Ground the answer in the candidate's concrete resume experience and metrics.",
    NEGOTIATION: "Stay factual and professional; anchor on market data and the candidate's value.",
    WRAP_UP: "Be concise; offer a thoughtful closing question if appropriate.",
}


def detect_mode(question: str, qtype: str = "", topic: str = "",
                recent: list[str] | None = None) -> str:
    """Pick the operating mode (composes phase detection). Never raises."""
    try:
        ph = _phase.detect_phase(question, qtype=qtype, topic=topic, recent=recent)
        return _PHASE_MODE.get(ph, GENERAL)
    except Exception:  # noqa: BLE001
        return GENERAL


def directive(mode: str) -> str:
    """Answer-shaping directive for a mode, or '' for GENERAL/unknown."""
    try:
        return _DIRECTIVE.get(mode or "", "")
    except Exception:  # noqa: BLE001
        return ""


class ModeTracker:
    """Per-session current mode with hysteresis: a switch requires the same new
    mode twice in a row so a single stray cue doesn't flip the whole session."""

    def __init__(self) -> None:
        self.mode = GENERAL
        self._pending: str | None = None

    def update(self, question: str, qtype: str = "", topic: str = "",
               recent: list[str] | None = None) -> str:
        try:
            new = detect_mode(question, qtype=qtype, topic=topic, recent=recent)
            if new == self.mode:
                self._pending = None
                return self.mode
            if new == self._pending:
                self.mode = new
                self._pending = None
            else:
                self._pending = new
            return self.mode
        except Exception:  # noqa: BLE001
            return self.mode


def for_tracker(tracker) -> ModeTracker:
    """Per-session ModeTracker attached to the context tracker (lazily)."""
    m = getattr(tracker, "_live_mode", None)
    if m is None:
        m = ModeTracker()
        try:
            tracker._live_mode = m
        except Exception:  # noqa: BLE001
            pass
    return m
