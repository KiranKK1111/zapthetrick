"""Planner — emits the Intent + execution plan onto the blackboard.

P0, must finish under ~250ms, so it stays non-blocking: intent UNDERSTANDING is
delegated to the model (the Persona adapts per turn from one comprehensive
system prompt) and the (LLM) Clarifier decides for itself whether a clarifying
question helps. No keyword classification or keyword gating happens here.
"""
from __future__ import annotations

from .. import pipeline
from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_INTENT, KEY_PLAN, KEY_QUESTION, Intent, Plan
from ..blackboard.scheduler import P0
from .base import Agent

class PlannerAgent(Agent):
    name = "planner"
    priority = P0
    expected_latency_ms = 200
    reads = frozenset({KEY_QUESTION})
    writes = frozenset({KEY_INTENT, KEY_PLAN})

    async def run(self, board: Blackboard) -> None:
        question = board.get(KEY_QUESTION, "")
        # Intent understanding is the model's job: the Persona adapts per turn
        # from one comprehensive system prompt, and the (LLM) Clarifier itself
        # decides whether a clarifying question helps — it declines on greetings
        # / acknowledgements / trivial turns. So we no longer keyword-classify
        # the intent or keyword-gate the Clarifier; always let it judge.
        label = pipeline.classify_intent(question)
        intent = Intent(
            type=label,
            topic="",
            urgency="normal",
            needs_clarification=True,
        )
        board.write(KEY_INTENT, intent, agent=self.name)

        # Phase-5 (ArchitectureVerdict): a REAL plan when the request is
        # multi-goal. The deterministic decomposer splits compound requests
        # ("build X, add Y, then document Z") into ordered sub-tasks; a simple
        # single-goal turn returns [] and keeps the legacy linear plan.
        # Flag-gated + fail-open → legacy behavior on any error.
        subtask_steps: list[str] = []
        try:
            from app.core.config_loader import cfg as _cfg
            if getattr(_cfg.orchestration, "planner_decompose", True):
                from app.orchestration.decompose import decompose
                subtask_steps = [t.text for t in decompose(question or "")]
        except Exception:  # noqa: BLE001
            subtask_steps = []

        if subtask_steps:
            # retrieve → each sub-goal in dependency order → ground.
            steps = ["retrieve", *subtask_steps, "ground"]
            plan = Plan(
                steps=steps,
                priorities=[P0] * len(steps),
                deadlines_ms=[500, *([4000] * len(subtask_steps)), 1000],
                parallel=False,
            )
        else:
            plan = Plan(
                steps=["retrieve", "respond", "ground"],
                priorities=[P0, P0, P0],
                deadlines_ms=[500, 4000, 1000],
                parallel=False,
            )
        board.write(KEY_PLAN, plan, agent=self.name)
