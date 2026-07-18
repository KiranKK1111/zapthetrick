"""Multi-model strategy selection (intelligent-model-routing R6).

`select_strategy(signals) -> "single" | "plan_exec_validate" | "consensus"`
chooses a workflow from the request signals (difficulty, task category, tool
needs). `MultiModelRunner` assigns each role to a capability-matched model via
the injected `route_request` and combines outputs through the EXISTING
`verified_answer` plumbing (R6.3/R6.4) — it never re-implements the
perceived-speed speculative race. Disabled / absent data → "single" (R6.5,
Property 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

SINGLE = "single"
PLAN_EXEC_VALIDATE = "plan_exec_validate"
CONSENSUS = "consensus"

# Roles per strategy (each assigned a capability-matched model).
_ROLES = {
    PLAN_EXEC_VALIDATE: ("planner", "executor", "validator"),
    CONSENSUS: ("member_a", "member_b", "judge"),
}


@dataclass
class StrategyDecision:
    strategy: str
    roles: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)


def select_strategy(signals: dict | None, *, enabled: bool = True) -> StrategyDecision:
    """Pick the workflow. Disabled / missing signals → single (Property 6/9)."""
    try:
        if not enabled or not signals:
            return StrategyDecision(SINGLE, (), ["disabled or no signals"])
        difficulty = str(signals.get("difficulty", "standard")).lower()
        category = str(signals.get("task_category", "general")).lower()

        # Expert, multi-step build/architecture/agentic work benefits from a
        # plan→execute→validate split.
        if difficulty == "expert" and category in (
                "agentic", "architecture", "coding"):
            return StrategyDecision(PLAN_EXEC_VALIDATE, _ROLES[PLAN_EXEC_VALIDATE],
                                    [f"expert {category} → plan/exec/validate"])
        # Hard reasoning/math benefits from consensus (sample + judge).
        if difficulty in ("hard", "expert") and category in ("reasoning", "math"):
            return StrategyDecision(CONSENSUS, _ROLES[CONSENSUS],
                                    [f"{difficulty} {category} → consensus"])
        return StrategyDecision(SINGLE, (), ["default single model"])
    except Exception:  # noqa: BLE001
        return StrategyDecision(SINGLE, (), ["strategy error → single"])


@dataclass
class MultiModelResult:
    answer: str
    strategy: str
    roles_used: dict
    degraded_to_single: bool = False


class MultiModelRunner:
    """Assigns each strategy role to a capability-matched model (via the injected
    `route_request`) and combines through the existing verification plumbing.

    `route_for(role, category) -> model_label` and `generate(role, model_label)
    -> str` are injected so this stays pure/testable and so the live path plugs
    in `route_request` + `verified_answer` without duplication. Any failure
    degrades to a single-model answer (R6.5 / error handling)."""

    def __init__(self,
                 route_for: Callable[[str, str], Awaitable[str]],
                 generate: Callable[[str, str], Awaitable[str]],
                 combine: Callable[[dict], str] | None = None):
        self._route_for = route_for
        self._generate = generate
        self._combine = combine or self._default_combine

    @staticmethod
    def _default_combine(role_outputs: dict) -> str:
        # The validator / judge output is the final answer when present, else
        # the executor's, else any.
        for key in ("validator", "judge", "executor", "member_b", "member_a"):
            if role_outputs.get(key):
                return role_outputs[key]
        return next(iter(role_outputs.values()), "")

    async def run(self, decision: StrategyDecision, category: str) -> MultiModelResult:
        if decision.strategy == SINGLE or not decision.roles:
            try:
                model = await self._route_for("single", category)
                ans = await self._generate("single", model)
                return MultiModelResult(ans, SINGLE, {"single": model})
            except Exception:  # noqa: BLE001
                return MultiModelResult("", SINGLE, {}, degraded_to_single=True)
        try:
            outputs: dict = {}
            models: dict = {}
            for role in decision.roles:
                model = await self._route_for(role, category)
                models[role] = model
                outputs[role] = await self._generate(role, model)
            return MultiModelResult(self._combine(outputs), decision.strategy, models)
        except Exception:  # noqa: BLE001 — degrade to single (R6.5)
            try:
                model = await self._route_for("single", category)
                ans = await self._generate("single", model)
                return MultiModelResult(ans, SINGLE, {"single": model},
                                        degraded_to_single=True)
            except Exception:  # noqa: BLE001
                return MultiModelResult("", SINGLE, {}, degraded_to_single=True)


__all__ = [
    "select_strategy", "StrategyDecision", "MultiModelRunner",
    "MultiModelResult", "SINGLE", "PLAN_EXEC_VALIDATE", "CONSENSUS",
]
