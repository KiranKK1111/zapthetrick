"""Priority + dataflow scheduler for agents on the blackboard.

Three priority lanes:
  P0 — REAL-TIME (must finish inside the latency budget)
  P1 — IMPROVEMENT (parallel, can be cancelled)
  P2 — BACKGROUND (post-response, no deadline)

The scheduler picks the highest-priority agent whose `reads` slots are
all present and runs it. P0 has a hard deadline; P1 has a soft grace
window; P2 is fire-and-forget after the user-facing path finishes.

Priority score (Architecture.md §5):
    priority = (1.0 / expected_latency_ms) * value_factor
where value_factor = 1.0 (P0), 0.5 (P1), 0.1 (P2), with bumps for
downstream-blocked agents and recent user preference.
"""
from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base import Agent
    from .board import Blackboard


# Priority lanes — also used as the `priority` attribute on [Agent].
P0 = 0   # real-time
P1 = 1   # improvement
P2 = 2   # background


_VALUE_FACTOR = {P0: 1.0, P1: 0.5, P2: 0.1}


@dataclass(order=True)
class _ScheduleEntry:
    """Heap entry. We sort by `-score` so higher scores pop first."""
    neg_score: float
    seq: int                         # tiebreaker, stable order
    agent: "Agent" = field(compare=False)
    deadline_ms: int = field(compare=False, default=0)


class PriorityScheduler:
    """Pulls ready agents off a heap and runs them concurrently per lane.

    The supervisor owns one of these per turn. Usage:

        sched = PriorityScheduler(board, deadlines={"intent": 250, ...})
        sched.add(planner_agent)
        sched.add(retriever_agent)
        ...
        await sched.run_p0_p1(latency_budget_ms=8000)
        # P2 agents are launched separately via run_p2_background()
    """

    def __init__(
        self,
        board: "Blackboard",
        *,
        deadlines_ms: dict[str, int] | None = None,
        budget: "object | None" = None,
        memory_pressure: "object | None" = None,
    ) -> None:
        self.board = board
        self.deadlines_ms = deadlines_ms or {}
        self._heap: list[_ScheduleEntry] = []
        self._seq = 0
        self._running: dict[str, asyncio.Task] = {}
        self._results: dict[str, object] = {}
        self._started_ms = int(time.time() * 1000)
        # P5 #1 latency budget: recompute each stage's deadline from the budget
        # actually remaining (upstream overrun squeezes downstream). Optional;
        # None → the static `deadlines_ms` table (today's behaviour).
        self._budget = budget
        # P5 #12 resource scheduler: shed optional lanes under memory pressure.
        self._mem = memory_pressure
        if self._mem is None:
            try:
                from .memory_pressure import controller as _mc
                self._mem = _mc
            except Exception:  # noqa: BLE001 — pressure gating is best-effort
                self._mem = None

    def _deadline_for(self, agent: "Agent", entry: "_ScheduleEntry",
                      fallback_ms: int) -> int:
        """Deadline to grant `agent` now. When a LatencyBudget is attached, the
        static per-stage deadline is clamped to the budget still remaining so a
        slow upstream stage automatically squeezes this one. Fail-open."""
        static = entry.deadline_ms or self.deadlines_ms.get(agent.name, 0) or fallback_ms
        if self._budget is None:
            return static
        try:
            self._budget.stage_deadlines_ms.setdefault(agent.name, float(static))
            return int(self._budget.deadline_for(agent.name))
        except Exception:  # noqa: BLE001
            return static

    def _admits(self, agent: "Agent") -> bool:
        """Memory-pressure gate: under high/critical pressure, optional lanes
        (P1/P2) are shed so the P0 real-time path keeps headroom. Fail-open."""
        if self._mem is None:
            return True
        try:
            return bool(self._mem.admits(agent.priority))
        except Exception:  # noqa: BLE001
            return True

    # ---- registration ------------------------------------------------
    def add(self, agent: "Agent", *, deadline_ms: int | None = None) -> None:
        """Enqueue an agent. Score is recomputed on each tick."""
        score = self._score(agent)
        self._seq += 1
        heapq.heappush(
            self._heap,
            _ScheduleEntry(
                neg_score=-score,
                seq=self._seq,
                agent=agent,
                deadline_ms=deadline_ms or self.deadlines_ms.get(agent.name, 0),
            ),
        )

    def _score(self, agent: "Agent") -> float:
        base = 1.0 / max(agent.expected_latency_ms, 1)
        score = base * _VALUE_FACTOR.get(agent.priority, 0.1)
        # +0.3 if other agents are blocked on this one's outputs.
        if any(self._depends_on(other, agent) for other in self._all_agents()):
            score += 0.3
        return score

    def _all_agents(self) -> list["Agent"]:
        return [e.agent for e in self._heap] + [
            a for a in self._running.values() if hasattr(a, "agent")  # type: ignore
        ]

    def _depends_on(self, dependent: "Agent", producer: "Agent") -> bool:
        return bool(dependent.reads & producer.writes)

    # ---- ready check -------------------------------------------------
    def _ready(self, agent: "Agent") -> bool:
        return all(self.board.has(k) or k == "question" and self.board.has("question")
                   for k in agent.reads)

    # ---- main loop ---------------------------------------------------
    async def run_p0_p1(self, *, latency_budget_ms: int) -> None:
        """Drive P0 and P1 agents until the deadline.

        P0 agents are awaited. P1 agents run as background tasks; any
        still-running at the deadline are cancelled.
        """
        hard_deadline = self._started_ms + latency_budget_ms

        while self._heap:
            now_ms = int(time.time() * 1000)
            if now_ms >= hard_deadline:
                break

            # Find the highest-scoring ready agent.
            ready_idx = next(
                (i for i, e in enumerate(self._heap) if self._ready(e.agent)),
                None,
            )
            if ready_idx is None:
                # Nothing ready — wait briefly for the board to advance.
                await asyncio.sleep(0.005)
                continue

            entry = self._heap.pop(ready_idx)
            heapq.heapify(self._heap)
            agent = entry.agent

            if agent.priority == P2:
                # Don't run P2 in the user-facing path.
                continue

            # Resource scheduler (P5 #12): shed optional lanes under pressure.
            if not self._admits(agent):
                continue

            task = asyncio.create_task(
                self._run_one(agent,
                              self._deadline_for(agent, entry, latency_budget_ms)),
                name=f"agent:{agent.name}",
            )
            self._running[agent.name] = task

            if agent.priority == P0:
                # Block on P0 so downstream agents can react.
                try:
                    await task
                except Exception:
                    pass

        # Wait briefly for P1 stragglers, then cancel.
        await self._wait_for_p1(hard_deadline)

    async def _run_one(self, agent: "Agent", deadline_ms: int) -> None:
        try:
            await asyncio.wait_for(
                agent.run(self.board),
                timeout=max(deadline_ms, 1) / 1000.0,
            )
        except asyncio.TimeoutError:
            # Agent missed its deadline — drop quietly.
            pass
        except Exception:
            # Agents must be resilient; one failure doesn't sink the turn.
            pass

    async def _wait_for_p1(self, hard_deadline_ms: int) -> None:
        remaining_s = max(0, hard_deadline_ms - int(time.time() * 1000)) / 1000.0
        if not self._running or remaining_s <= 0:
            for task in self._running.values():
                task.cancel()
            return
        await asyncio.wait(
            self._running.values(),
            timeout=remaining_s,
            return_when=asyncio.ALL_COMPLETED,
        )
        for name, task in list(self._running.items()):
            if not task.done():
                task.cancel()

    # ---- P2 fire-and-forget -----------------------------------------
    def run_p2_background(self, agents: "list[Agent]") -> None:
        """Schedule P2 agents after the user-facing path finishes.

        These run with no deadline; failures are silent.
        """
        for agent in agents:
            # Under memory pressure, background (P2) work is the first to be shed.
            if not self._admits(agent):
                continue
            asyncio.create_task(
                self._run_one(agent, deadline_ms=60_000),
                name=f"agent-bg:{agent.name}",
            )
