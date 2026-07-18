"""
Interruption / self-correction detection (live-conversational-intelligence R13).

Detects when the interviewer abandons or supersedes the current question
("actually, before that…", "scratch that, explain RabbitMQ instead") so the live
loop can cancel the in-flight answer's `qid`(s) and switch to the new question.
Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core import lexicons

_INTERRUPT_CUES = lexicons.LIVE_INTERRUPT_CUES
_CORRECTION_CUES = lexicons.LIVE_INTERRUPT_CORRECTION_CUES


@dataclass
class InterruptSignal:
    interrupted: bool
    self_correction: bool


def detect(text: str) -> InterruptSignal:
    """Detect an interruption / self-correction in the utterance. Never raises."""
    try:
        t = (text or "").strip().lower()
        if not t:
            return InterruptSignal(False, False)
        interrupted = any(c in t for c in _INTERRUPT_CUES)
        correction = any(c in t for c in _CORRECTION_CUES)
        return InterruptSignal(interrupted=interrupted or correction,
                               self_correction=correction)
    except Exception:  # noqa: BLE001
        return InterruptSignal(False, False)


def should_cancel(text: str) -> bool:
    """True when the in-flight answer(s) should be cancelled for this utterance."""
    return detect(text).interrupted
