"""Latency observatory (perceived-speed R16).

Records per-stage durations and TTFT for each request so the real bottleneck is
visible (dev build). When a stage exceeds its configured budget, it is flagged
as the request's bottleneck (R16.4). In-memory + bounded; never affects request
handling (read-only telemetry).
"""
from __future__ import annotations

from collections import OrderedDict

# The pipeline stages we time (R16.1).
STAGES = ("submit", "network", "routing", "retrieval", "provider", "streaming", "render")


class LatencyObservatory:
    def __init__(self, cap: int = 500) -> None:
        self._reqs: "OrderedDict[str, dict]" = OrderedDict()
        self._cap = max(1, cap)

    def _rec(self, request_id: str) -> dict:
        r = self._reqs.get(request_id)
        if r is None:
            r = {"stages": {}, "ttft_ms": None}
            self._reqs[request_id] = r
            self._reqs.move_to_end(request_id)
            while len(self._reqs) > self._cap:
                self._reqs.popitem(last=False)
        return r

    def record(self, request_id: str, stage: str, ms: float) -> None:
        if stage not in STAGES:
            return
        self._rec(request_id)["stages"][stage] = float(ms)

    def record_ttft(self, request_id: str, ms: float) -> None:
        self._rec(request_id)["ttft_ms"] = float(ms)

    def bottleneck(self, request_id: str, budgets: dict | None = None) -> str | None:
        """The stage exceeding its budget by the largest margin (R16.4), or the
        slowest stage when no budgets are given."""
        r = self._reqs.get(request_id)
        if not r or not r["stages"]:
            return None
        stages = r["stages"]
        if budgets:
            over = {s: stages[s] - budgets[s]
                    for s in stages if s in budgets and stages[s] > budgets[s]}
            if over:
                return max(over, key=over.get)
            return None
        return max(stages, key=stages.get)

    def report(self, request_id: str, budgets: dict | None = None) -> dict | None:
        r = self._reqs.get(request_id)
        if r is None:
            return None
        return {
            "request_id": request_id,
            "stages": dict(r["stages"]),
            "ttft_ms": r["ttft_ms"],
            "bottleneck": self.bottleneck(request_id, budgets),
        }

    def reset(self) -> None:
        self._reqs.clear()


# Process-wide observatory (dev telemetry).
observatory = LatencyObservatory()

__all__ = ["LatencyObservatory", "observatory", "STAGES"]
