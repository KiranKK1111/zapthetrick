"""Latency budget manager (roadmap Phase 5 #1).

The `PriorityScheduler` (see `scheduler.py`) ships STATIC per-stage
`deadlines_ms` — intent 250, plan 200, retrieve 500, first_token 1500, total
8000. That is a fixed table: if the intent stage burns 900ms instead of 250ms,
the downstream stages still each believe they have their full static slice, and
the turn quietly blows the 8000ms total.

`LatencyBudget` turns that table into a REAL budget: it tracks wall-clock spend
against the total, and `deadline_for(stage)` returns the SMALLER of the stage's
static allowance and the budget actually remaining. So when an upstream stage
overruns, every downstream stage is squeezed automatically and the total holds.

Deterministic + injectable clock (tests advance time), in-process, fail-open —
any error yields the static deadline (today's behaviour).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


def _now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass
class LatencyBudget:
    """Downstream-aware latency budget for one turn.

    `total_ms` is the whole-turn ceiling; `stage_deadlines_ms` are the static
    per-stage allowances. As stages run, elapsed time is charged against the
    total and `deadline_for` shrinks the remaining stages' deadlines so the
    total is honoured even when an earlier stage overran.
    """

    total_ms: float
    stage_deadlines_ms: dict[str, float] = field(default_factory=dict)
    now: Callable[[], float] = _now_ms
    _start: float = field(default=0.0, init=False)
    _consumed: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._start = self.now()

    # ── clock ─────────────────────────────────────────────────────────────
    def elapsed_ms(self) -> float:
        return max(0.0, self.now() - self._start)

    def remaining_ms(self) -> float:
        """Budget still available for all not-yet-run downstream work."""
        return max(0.0, self.total_ms - self.elapsed_ms())

    def over_budget(self) -> bool:
        return self.elapsed_ms() >= self.total_ms

    # ── deadlines ─────────────────────────────────────────────────────────
    def deadline_for(self, stage: str, *, floor_ms: float = 1.0) -> float:
        """Deadline to grant `stage` RIGHT NOW.

        = min(static stage allowance, budget remaining). If the budget is
        already blown, returns `floor_ms` (a tiny non-zero grant) rather than 0
        so a caller that treats 0 as "no timeout" never accidentally runs
        unbounded. Fail-open: unknown stage → the whole remaining budget.
        """
        try:
            remaining = self.remaining_ms()
            static = self.stage_deadlines_ms.get(stage)
            grant = remaining if static is None else min(float(static), remaining)
            return max(floor_ms, grant)
        except Exception:  # noqa: BLE001
            return float(self.stage_deadlines_ms.get(stage, self.total_ms) or floor_ms)

    # ── accounting ────────────────────────────────────────────────────────
    def consume(self, stage: str, ms: float) -> None:
        """Charge `ms` to `stage` (bookkeeping / observability)."""
        try:
            self._consumed[stage] = self._consumed.get(stage, 0.0) + max(0.0, float(ms))
        except Exception:  # noqa: BLE001
            pass

    def consumed(self, stage: str | None = None) -> float:
        if stage is not None:
            return self._consumed.get(stage, 0.0)
        return sum(self._consumed.values())

    @contextmanager
    def stage(self, name: str):
        """Time a stage; the elapsed wall-clock is charged to it on exit so the
        remaining-budget view is accurate for whatever runs next."""
        t0 = self.now()
        try:
            yield self.deadline_for(name)
        finally:
            self.consume(name, self.now() - t0)

    def snapshot(self) -> dict:
        return {
            "total_ms": self.total_ms,
            "elapsed_ms": round(self.elapsed_ms(), 1),
            "remaining_ms": round(self.remaining_ms(), 1),
            "over_budget": self.over_budget(),
            "consumed": {k: round(v, 1) for k, v in self._consumed.items()},
        }


def default_budget(now: Callable[[], float] = _now_ms,
                   deadlines_ms: dict[str, float] | None = None) -> LatencyBudget:
    """Build a budget from the standard per-stage deadlines. Callers with a
    config-derived table (the supervisor, which already depends on `core`) pass
    `deadlines_ms` — keeping this blackboard module free of a cross-package
    import to `core` (import-boundary guardrail)."""
    total = 8000.0
    stages: dict[str, float] = {
        "intent": 250.0, "plan": 200.0, "retrieve": 500.0, "first_token": 1500.0,
    }
    if deadlines_ms:
        total = float(deadlines_ms.get("total", total) or total)
        for k, v in deadlines_ms.items():
            if k != "total":
                stages[k] = float(v)
    return LatencyBudget(total_ms=total, stage_deadlines_ms=stages, now=now)


__all__ = ["LatencyBudget", "default_budget"]
