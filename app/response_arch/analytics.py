"""Stream analytics — TTFMU / first-code / artifact-ready (roadmap Phase 6 #21).

The latency observatory already records per-stage TTFT, but the specific
streaming milestones the roadmap calls out — **time to first *meaningful*
update** (the first token/block the user can actually read, not an ``ack``),
**time to the first code block**, and **time to the first artifact ready** — were
never instrumented. :class:`StreamAnalytics` records those marks from a single
monotonic clock and computes the derived durations, fail-open.

It is a passive recorder: the streaming loop calls :meth:`mark` at the moments it
already knows about (first token, a closed code block, an artifact emission), and
:meth:`metrics` returns a compact dict suitable for an ``analytics`` telemetry
frame or the envelope ``meta``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

# canonical mark names
SUBMIT = "submit"
FIRST_MEANINGFUL = "first_meaningful_update"   # TTFMU
FIRST_CODE = "first_code_block"
ARTIFACT_READY = "artifact_ready"
DONE = "done"


@dataclass
class StreamAnalytics:
    clock: Callable[[], float] = time.monotonic
    _t0: float | None = None
    _marks: dict[str, float] = field(default_factory=dict)

    def start(self) -> "StreamAnalytics":
        self._t0 = self.clock()
        self._marks[SUBMIT] = self._t0
        return self

    def mark(self, name: str) -> None:
        """Record a milestone (first occurrence wins — idempotent)."""
        try:
            if self._t0 is None:
                self.start()
            if name and name not in self._marks:
                self._marks[name] = self.clock()
        except Exception:  # noqa: BLE001
            pass

    def _ms(self, name: str) -> int | None:
        t = self._marks.get(name)
        if t is None or self._t0 is None:
            return None
        return int(round((t - self._t0) * 1000.0))

    def metrics(self) -> dict:
        """Derived durations in ms (absent marks omitted)."""
        out = {
            "ttfmu_ms": self._ms(FIRST_MEANINGFUL),
            "first_code_ms": self._ms(FIRST_CODE),
            "artifact_ready_ms": self._ms(ARTIFACT_READY),
            "total_ms": self._ms(DONE),
        }
        return {k: v for k, v in out.items() if v is not None}

    def as_frame(self) -> dict:
        return self.metrics()


__all__ = [
    "StreamAnalytics", "SUBMIT", "FIRST_MEANINGFUL", "FIRST_CODE",
    "ARTIFACT_READY", "DONE",
]
