"""
Interview-phase detection (live-conversational-intelligence R7).

Deterministic-first: classifies the current Interview_Phase from the question,
its type/topic, and recent questions using cue lexicons — no LLM call. The
detected phase feeds answer-strategy selection (R8) and is surfaced as additive
`meta.phase`. Fail-open: on any error returns the neutral default.
"""
from __future__ import annotations

from app.core import lexicons

INTRODUCTION = "introduction"
RESUME_DISCUSSION = "resume_discussion"
TECHNICAL_SCREENING = "technical_screening"
SYSTEM_DESIGN = "system_design"
CODING = "coding"
BEHAVIORAL = "behavioral"
HR = "hr"
CLOSING = "closing"

PHASES = {
    INTRODUCTION, RESUME_DISCUSSION, TECHNICAL_SCREENING, SYSTEM_DESIGN,
    CODING, BEHAVIORAL, HR, CLOSING,
}

_INTRO_CUES = lexicons.LIVE_PHASE_INTRO_CUES
_RESUME_CUES = lexicons.LIVE_PHASE_RESUME_CUES
_DESIGN_CUES = lexicons.LIVE_PHASE_DESIGN_CUES
_CODING_CUES = lexicons.LIVE_PHASE_CODING_CUES
_BEHAVIORAL_CUES = lexicons.LIVE_PHASE_BEHAVIORAL_CUES
_HR_CUES = lexicons.LIVE_PHASE_HR_CUES
_CLOSING_CUES = lexicons.LIVE_PHASE_CLOSING_CUES


def detect_phase(
    question: str,
    qtype: str = "",
    topic: str = "",
    recent: list[str] | None = None,
    summary: str = "",
) -> str:
    """Detect the current Interview_Phase (deterministic, no LLM call)."""
    try:
        t = (question or "").lower()
        qt = (qtype or "").lower()

        if any(c in t for c in _CLOSING_CUES):
            return CLOSING
        if any(c in t for c in _HR_CUES):
            return HR
        if any(c in t for c in _INTRO_CUES):
            return INTRODUCTION
        if qt == "behavioral" or any(c in t for c in _BEHAVIORAL_CUES):
            return BEHAVIORAL
        if qt == "coding" or any(c in t for c in _CODING_CUES):
            return CODING
        if any(c in t for c in _DESIGN_CUES) or "system" in (topic or "").lower():
            return SYSTEM_DESIGN
        if any(c in t for c in _RESUME_CUES):
            return RESUME_DISCUSSION
        return TECHNICAL_SCREENING
    except Exception:  # noqa: BLE001
        return TECHNICAL_SCREENING
