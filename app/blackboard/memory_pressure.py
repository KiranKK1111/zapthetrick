"""Resource scheduler — memory-pressure controller (roadmap Phase 5 #12).

The `PriorityScheduler` already orders work by priority lane (P0 real-time > P1
improvement > P2 background). What it lacked was a resource signal: under memory
pressure the right thing is to SHED the optional lanes (drop P2 entirely, then
P1) so the P0 real-time path always has headroom, instead of letting speculative
/ background work push the process into swap or the OOM killer.

`MemoryPressure.level()` samples RSS-vs-available (psutil when present, a cheap
`resource`/`gc` fallback otherwise) and buckets it: ok / elevated / high /
critical. `admits(priority)` is the gate the scheduler consults before launching
an agent. Sampled + cached (a short TTL) so it's free to call per dispatch.

Deterministic-enough + fully fail-open: any probe error reports `ok` (admit
everything = today's behaviour).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

OK = "ok"
ELEVATED = "elevated"
HIGH = "high"
CRITICAL = "critical"

_ORDER = {OK: 0, ELEVATED: 1, HIGH: 2, CRITICAL: 3}

# Priority lanes (mirror scheduler.P0/P1/P2 without importing to avoid a cycle).
_P0, _P1, _P2 = 0, 1, 2


def _system_used_fraction() -> float | None:
    """Fraction (0..1) of system memory in use, or None when unknowable."""
    try:
        import psutil
        return float(psutil.virtual_memory().percent) / 100.0
    except Exception:  # noqa: BLE001
        pass
    try:
        # POSIX fallback: RSS vs a rough total from resource limits is unreliable,
        # so we only report when /proc-style data is present.
        import os
        if hasattr(os, "sysconf") and os.sysconf_names.get("SC_PAGE_SIZE") \
                and os.sysconf_names.get("SC_PHYS_PAGES"):
            page = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES") * page
            avail = os.sysconf("SC_AVPHYS_PAGES") * page \
                if os.sysconf_names.get("SC_AVPHYS_PAGES") else None
            if total and avail is not None:
                return max(0.0, min(1.0, 1.0 - (avail / total)))
    except Exception:  # noqa: BLE001
        pass
    return None


@dataclass
class Thresholds:
    elevated: float = 0.75
    high: float = 0.85
    critical: float = 0.93


class MemoryPressure:
    """Sampled memory-pressure gate for the scheduler."""

    def __init__(
        self,
        *,
        sample: Callable[[], float | None] = _system_used_fraction,
        thresholds: Thresholds | None = None,
        ttl_s: float = 0.5,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sample = sample
        self._th = thresholds or Thresholds()
        self._ttl = ttl_s
        self._now = now
        self._cached_at = -1e9
        self._cached_frac: float | None = None

    def used_fraction(self) -> float | None:
        now = self._now()
        if now - self._cached_at <= self._ttl:
            return self._cached_frac
        try:
            self._cached_frac = self._sample()
        except Exception:  # noqa: BLE001
            self._cached_frac = None
        self._cached_at = now
        return self._cached_frac

    def level(self) -> str:
        frac = self.used_fraction()
        if frac is None:
            return OK
        if frac >= self._th.critical:
            return CRITICAL
        if frac >= self._th.high:
            return HIGH
        if frac >= self._th.elevated:
            return ELEVATED
        return OK

    def admits(self, priority: int) -> bool:
        """Whether an agent in `priority` lane may launch under current pressure.

        - critical → only P0 (real-time) runs; everything optional is shed.
        - high     → P0 + P1; background P2 is shed.
        - elevated → everything, but the caller may throttle concurrency.
        - ok       → everything.
        Fail-open: on any error, admit.
        """
        try:
            lvl = _ORDER[self.level()]
            if lvl >= _ORDER[CRITICAL]:
                return priority <= _P0
            if lvl >= _ORDER[HIGH]:
                return priority <= _P1
            return True
        except Exception:  # noqa: BLE001
            return True

    def sheds_background(self) -> bool:
        return _ORDER[self.level()] >= _ORDER[HIGH]

    def snapshot(self) -> dict:
        return {"level": self.level(), "used_fraction": self.used_fraction()}


# Process-wide controller shared by the scheduler. Always available + fully
# fail-open (unreadable memory → admits everything), so it needs no config gate;
# keeping config-reads out of this module avoids a blackboard→core import edge
# (import-boundary guardrail). Disable by constructing the scheduler with
# `memory_pressure=` explicitly set to a permissive/None controller.
controller = MemoryPressure()


__all__ = [
    "MemoryPressure", "Thresholds", "controller",
    "OK", "ELEVATED", "HIGH", "CRITICAL",
]
