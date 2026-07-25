"""Drift-proof audio transport — PCM sequence tracking (vNext §4.7, Component C).

Live audio is a continuous PCM stream over the WebSocket; packets can be lost,
reordered, or duplicated by the network, and clock drift between the sender and
receiver accumulates over a 2-hour session. The client owns a ~30 s ring buffer +
arrival-jitter re-timing (FE); this is the BACKEND half — a per-session
`SequenceTracker` that reads the PCM SEQUENCE NUMBER on each chunk and reports
gaps / reorders / duplicates + a jitter estimate, so the pipeline can surface a
degraded-transport chip and the STT layer can compensate for a gap rather than
transcribe a discontinuity as garbage.

Pure + fail-open — bad input never raises, it's just counted or ignored. Bounded
memory (a small recent-seq window) so a long session can't grow it. Flag-gated by
`live.audio_watchdog` at the call site; the tracker itself is always-safe to run.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

OK = "ok"
GAP = "gap"
REORDER = "reorder"
DUPLICATE = "duplicate"

# How many recent sequence numbers to remember for duplicate/reorder detection.
_WINDOW = 256


@dataclass
class SequenceTracker:
    """Per-session PCM sequence bookkeeping. `observe(seq)` classifies each chunk
    and updates counters; `stats()` is the transport-health snapshot."""
    expected: int | None = None
    received: int = 0
    gaps: int = 0             # count of MISSING packets (not gap events)
    reorders: int = 0
    duplicates: int = 0
    last_arrival: float | None = None
    jitter_ms: float = 0.0    # EWMA of |inter-arrival − nominal| (ms)
    _recent: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    _recent_set: set = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def observe(self, seq: int, *, arrival_s: float | None = None,
                nominal_ms: float | None = None) -> str:
        """Classify one chunk by its sequence number. Returns OK / GAP / REORDER /
        DUPLICATE. `arrival_s` + `nominal_ms` (expected inter-chunk interval) feed
        the jitter EWMA. Never raises."""
        try:
            with self._lock:
                seq = int(seq)
                self.received += 1
                self._jitter(arrival_s, nominal_ms)
                if self.expected is None:
                    self.expected = seq + 1
                    self._remember(seq)
                    return OK
                if seq == self.expected:
                    self.expected = seq + 1
                    self._remember(seq)
                    return OK
                if seq > self.expected:
                    self.gaps += (seq - self.expected)   # this many were skipped
                    self.expected = seq + 1
                    self._remember(seq)
                    return GAP
                # seq < expected → a late packet: duplicate if we've seen it.
                if seq in self._recent_set:
                    self.duplicates += 1
                    return DUPLICATE
                self.reorders += 1
                self._remember(seq)
                return REORDER
        except (TypeError, ValueError):
            return OK          # non-numeric seq → ignore, don't corrupt state
        except Exception:  # noqa: BLE001
            return OK

    def _remember(self, seq: int) -> None:
        if len(self._recent) == self._recent.maxlen and self._recent:
            self._recent_set.discard(self._recent[0])
        self._recent.append(seq)
        self._recent_set.add(seq)

    def _jitter(self, arrival_s: float | None, nominal_ms: float | None) -> None:
        if arrival_s is None:
            return
        if self.last_arrival is not None and nominal_ms:
            delta_ms = abs((arrival_s - self.last_arrival) * 1000.0 - nominal_ms)
            # EWMA (alpha 0.2) — a smoothed jitter estimate.
            self.jitter_ms = 0.8 * self.jitter_ms + 0.2 * delta_ms
        self.last_arrival = arrival_s

    def loss_rate(self) -> float:
        """Fraction of the stream that went missing (0..1)."""
        total = self.received + self.gaps
        return (self.gaps / total) if total > 0 else 0.0

    def degraded(self, *, loss_threshold: float = 0.05) -> bool:
        """Transport looks unhealthy — enough loss to warrant a chip."""
        return self.loss_rate() >= loss_threshold

    def stats(self) -> dict:
        return {"received": self.received, "gaps": self.gaps,
                "reorders": self.reorders, "duplicates": self.duplicates,
                "loss_rate": round(self.loss_rate(), 4),
                "jitter_ms": round(self.jitter_ms, 2)}


_LOCK = threading.RLock()
_TRACKERS: dict[str, SequenceTracker] = {}


def tracker(session_id: str) -> SequenceTracker:
    with _LOCK:
        t = _TRACKERS.get(session_id)
        if t is None:
            t = SequenceTracker()
            _TRACKERS[session_id] = t
        return t


def observe(session_id: str, seq: int, *, arrival_s: float | None = None,
            nominal_ms: float | None = None) -> str:
    """WS-layer entry: classify a PCM chunk for `session_id`. Never raises."""
    try:
        return tracker(session_id).observe(
            seq, arrival_s=arrival_s, nominal_ms=nominal_ms)
    except Exception:  # noqa: BLE001
        return OK


def forget_session(session_id: str) -> None:
    with _LOCK:
        _TRACKERS.pop(session_id, None)


def reset_for_tests() -> None:
    with _LOCK:
        _TRACKERS.clear()


__all__ = ["SequenceTracker", "OK", "GAP", "REORDER", "DUPLICATE",
           "tracker", "observe", "forget_session", "reset_for_tests"]
