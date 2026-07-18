"""Idle-time background work (roadmap Phase 5 #10).

After a turn's user-facing path finishes, the process is usually idle for
seconds-to-minutes before the next request. That idle time is free capacity for
non-urgent upkeep — summarising a long conversation, embedding new artifacts for
later retrieval, re-verifying a low-confidence answer — provided it NEVER touches
the hot path and yields the moment real work arrives.

`IdleScheduler` runs registered idle jobs one-at-a-time (never a thundering herd),
each guarded, each skipped under memory pressure (P5 #12), with a wall-clock
budget so a slow job can't monopolise the loop. Jobs are injected (callables), so
the scheduler carries no heavy deps and is fully testable.

Off unless enabled; fail-open everywhere.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Standard idle-job kinds (advisory labels).
SUMMARIZE = "summarize"
EMBED = "embed"
VERIFY = "verify"


def idle_enabled() -> bool:
    """`cfg.perceived.idle_work` — enabling default True (post-response only)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "perceived", None), "idle_work", True))
    except Exception:  # noqa: BLE001
        return True


@dataclass
class IdleJob:
    name: str
    run: Callable[[], Awaitable | object]     # sync or async, no args
    kind: str = SUMMARIZE
    max_ms: float = 5_000.0                    # per-job soft ceiling


@dataclass
class IdleReport:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    shed_for_pressure: bool = False


class IdleScheduler:
    """Runs idle jobs serially, best-effort, under a memory-pressure gate."""

    def __init__(self, *, memory=None) -> None:
        self._jobs: list[IdleJob] = []
        # The memory-pressure gate is INJECTED (the scheduler/supervisor, which
        # already depends on the blackboard, passes the controller) so this
        # perceived-package module carries no import to `blackboard`
        # (import-boundary guardrail). None → no shedding (fail-open: run jobs).
        self._mem = memory

    def register(self, job: IdleJob) -> None:
        self._jobs.append(job)

    def clear(self) -> None:
        self._jobs.clear()

    def _mem_ok(self) -> bool:
        if self._mem is None:
            return True
        try:
            # Idle work is background (P2); shed it exactly when P2 is shed.
            return not bool(self._mem.sheds_background())
        except Exception:  # noqa: BLE001
            return True

    async def run_once(self, *, budget_ms: float = 30_000.0) -> IdleReport:
        """Run all registered jobs once, serially. Skips everything under memory
        pressure; stops early when the overall `budget_ms` is spent."""
        report = IdleReport()
        if not idle_enabled() or not self._jobs:
            return report
        if not self._mem_ok():
            report.shed_for_pressure = True
            report.skipped = [j.name for j in self._jobs]
            return report
        start = time.monotonic()
        for job in list(self._jobs):
            if (time.monotonic() - start) * 1000.0 >= budget_ms:
                report.skipped.append(job.name)
                continue
            if not self._mem_ok():                 # pressure can rise mid-run
                report.shed_for_pressure = True
                report.skipped.append(job.name)
                continue
            try:
                res = job.run()
                if asyncio.iscoroutine(res) or asyncio.isfuture(res):
                    await asyncio.wait_for(res, timeout=max(job.max_ms, 1) / 1000.0)
                report.ran.append(job.name)
            except asyncio.TimeoutError:
                report.failed.append(job.name)
                log.info("idle job %s timed out", job.name)
            except Exception as exc:  # noqa: BLE001 — idle work is best-effort
                report.failed.append(job.name)
                log.info("idle job %s failed: %s", job.name, exc)
        return report

    def spawn(self, *, budget_ms: float = 30_000.0) -> "asyncio.Task | None":
        """Fire-and-forget the idle pass (post-response). Never blocks the caller."""
        if not idle_enabled() or not self._jobs:
            return None
        try:
            asyncio.get_running_loop()             # no loop → don't build a coro
        except RuntimeError:
            return None
        try:
            return asyncio.create_task(self.run_once(budget_ms=budget_ms),
                                       name="idle-work")
        except RuntimeError:                       # pragma: no cover
            return None


# Process-wide idle scheduler.
scheduler = IdleScheduler()


__all__ = [
    "IdleJob", "IdleReport", "IdleScheduler", "scheduler", "idle_enabled",
    "SUMMARIZE", "EMBED", "VERIFY",
]
