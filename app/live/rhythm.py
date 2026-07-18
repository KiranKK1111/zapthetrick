"""Conversation Rhythm / Energy / Fatigue model (roadmap Phase 2 #31).

Tracks the interview's *pacing* from inter-question gaps and volume: rapid-fire
vs measured, and interviewer fatigue/saturation as the session runs long. Lets
the planner adapt (shorter answers under rapid-fire; prep a pace change when
saturation sets in). Deterministic (caller supplies the measured gap so tests
stay time-free); per-session registry.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

_WINDOW = 5
_RAPID_FIRE_S = 8.0     # median gap below this ⇒ rapid-fire
_SLOW_S = 45.0          # median gap above this ⇒ slow/measured
_FATIGUE_COUNT = 25     # questions after which saturation is likely
_FATIGUE_MINUTES = 40   # session minutes after which saturation is likely


@dataclass
class RhythmTracker:
    gaps: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    count: int = 0
    elapsed_seconds: float = 0.0
    _last_ts: float | None = None

    def observe(self, gap_seconds: float | None = None) -> None:
        """Record one question. `gap_seconds` = seconds since the previous
        question (None for the first). Deterministic — caller supplies the gap."""
        try:
            self.count += 1
            if gap_seconds is not None and gap_seconds >= 0:
                self.gaps.append(float(gap_seconds))
                self.elapsed_seconds += float(gap_seconds)
        except Exception:  # noqa: BLE001
            pass

    def observe_now(self) -> None:
        """Live-path convenience: derive the gap from a monotonic clock, so the
        pipeline needs no timing plumbing. Caps absurd gaps (long pauses) so an
        idle stretch doesn't distort cadence."""
        try:
            now = time.monotonic()
            gap = None if self._last_ts is None else min(now - self._last_ts, 300.0)
            self._last_ts = now
            self.observe(gap)
        except Exception:  # noqa: BLE001
            pass

    def _median_gap(self) -> float | None:
        if not self.gaps:
            return None
        s = sorted(self.gaps)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def cadence(self) -> str:
        m = self._median_gap()
        if m is None:
            return "unknown"
        if m < _RAPID_FIRE_S:
            return "rapid_fire"
        if m > _SLOW_S:
            return "slow"
        return "steady"

    def is_rapid_fire(self) -> bool:
        return self.cadence() == "rapid_fire"

    def fatigue(self) -> float:
        """0..1 saturation estimate from question count and elapsed time."""
        by_count = min(1.0, self.count / _FATIGUE_COUNT)
        by_time = min(1.0, (self.elapsed_seconds / 60.0) / _FATIGUE_MINUTES)
        return round(max(by_count, by_time), 3)

    def snapshot(self) -> dict:
        return {
            "cadence": self.cadence(),
            "rapid_fire": self.is_rapid_fire(),
            "fatigue": self.fatigue(),
            "questions": self.count,
        }


_trackers: dict[str, RhythmTracker] = {}


def get_rhythm(session_id: str) -> RhythmTracker:
    t = _trackers.get(session_id)
    if t is None:
        t = RhythmTracker()
        _trackers[session_id] = t
    return t


def forget_session(session_id: str) -> None:
    _trackers.pop(session_id, None)


__all__ = ["RhythmTracker", "get_rhythm", "forget_session"]
