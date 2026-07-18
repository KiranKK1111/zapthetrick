"""Per-session interview-phase progression tracker (roadmap Phase 2 #3/#14).

`phase.py::detect_phase` is stateless — it classifies ONE utterance. This adds
the session-level *progression* on top: it smooths noisy per-utterance detections
(one stray "salary" cue shouldn't jump the whole interview to HR), records the
committed phase history + transitions, and estimates how far through the
interview we are. Feeds trajectory awareness ("we're near the end") and lets the
planner reason over the arc, not just the current question.

Deterministic, fail-open, per-session (registry mirrors state_machine.py).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.live import phase as _phase

# Canonical forward ordering of an interview. Progress = ordinal / (n-1).
PHASE_ORDER: list[str] = [
    _phase.INTRODUCTION,
    _phase.RESUME_DISCUSSION,
    _phase.TECHNICAL_SCREENING,
    _phase.CODING,
    _phase.SYSTEM_DESIGN,
    _phase.BEHAVIORAL,
    _phase.HR,
    _phase.CLOSING,
]
_ORDINAL = {p: i for i, p in enumerate(PHASE_ORDER)}
_LATE = {_phase.HR, _phase.CLOSING}

# How many recent detections to smooth over, and how many must agree to commit a
# transition (hysteresis — resists single-utterance noise).
_WINDOW = 3
_COMMIT = 2


@dataclass
class PhaseTracker:
    window: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    current: str = _phase.INTRODUCTION
    history: list[str] = field(default_factory=list)
    transitions: list[tuple[str, str]] = field(default_factory=list)

    def observe(self, detected: str) -> str:
        """Feed one per-utterance detected phase; return the (smoothed) current
        phase. Only commits a transition when >= _COMMIT of the recent window
        agree on a new phase, so a lone stray cue can't move the interview."""
        try:
            if detected not in _phase.PHASES:
                return self.current
            self.window.append(detected)
            if not self.history:
                # First real observation seeds the current phase.
                self.current = detected
                self.history.append(detected)
                return self.current
            # Count support for `detected` in the smoothing window.
            support = sum(1 for p in self.window if p == detected)
            if detected != self.current and support >= _COMMIT:
                self.transitions.append((self.current, detected))
                self.current = detected
                self.history.append(detected)
            return self.current
        except Exception:  # noqa: BLE001 — tracking must never break a turn
            return self.current

    def progress(self) -> float:
        """0.0 (intro) .. 1.0 (closing), from the current phase's ordinal."""
        n = len(PHASE_ORDER) - 1
        return round(_ORDINAL.get(self.current, 0) / n, 3) if n else 0.0

    def predict_next(self) -> str | None:
        """Forward trajectory (Phase 2 #14): the most likely NEXT phase — the
        next stage in the canonical order the interview hasn't settled into yet.
        Returns None at the final phase (closing)."""
        try:
            cur = _ORDINAL.get(self.current, 0)
            seen = {p for p in self.history}
            # Prefer the next unvisited forward phase; else simply the next in order.
            for p in PHASE_ORDER[cur + 1:]:
                if p not in seen:
                    return p
            return PHASE_ORDER[cur + 1] if cur + 1 < len(PHASE_ORDER) else None
        except Exception:  # noqa: BLE001
            return None

    def is_late_stage(self) -> bool:
        """True once we've reached HR/closing — time to prep negotiation/wrap-up."""
        return self.current in _LATE

    def distinct_phases(self) -> int:
        return len(dict.fromkeys(self.history))

    def snapshot(self) -> dict:
        return {
            "phase": self.current,
            "phase_progress": self.progress(),
            "phase_history": list(dict.fromkeys(self.history)),
            "phase_transitions": len(self.transitions),
            "late_stage": self.is_late_stage(),
            "predicted_next": self.predict_next(),
        }


_trackers: dict[str, PhaseTracker] = {}


def get_phase_tracker(session_id: str) -> PhaseTracker:
    t = _trackers.get(session_id)
    if t is None:
        t = PhaseTracker()
        _trackers[session_id] = t
    return t


def forget_session(session_id: str) -> None:
    _trackers.pop(session_id, None)


__all__ = [
    "PhaseTracker", "PHASE_ORDER", "get_phase_tracker", "forget_session",
]
