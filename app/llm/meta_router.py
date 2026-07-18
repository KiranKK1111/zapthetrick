"""Meta-router (intelligent-model-routing R7).

`decide(signals) -> RoutingDecision` unifies the TaskClassifier + StrategySelector
+ capability/tool requirements into ONE coherent routing decision, then
**delegates the final model+key pick to `route_request`** (R7.2) — it never
duplicates the penalty/headroom/fallback logic. It consumes the perceived-speed
latency signals and the evaluation Aggregate_Confidence passed in `signals`
(R7.3). Disabled → today's `route_request` path unchanged (R7.4, Property 6/9).

`decide` is deterministic + synchronous (no LLM call, Property 9); the caller
uses the decision to parametrize `route_request` and, for a multi-model
strategy, drive `MultiModelRunner` over the existing `verified_answer` plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.task_class import classify_task
from app.llm.strategy import select_strategy, SINGLE
from app.llm.escalation import escalation_chain


@dataclass
class RoutingDecision:
    task_category: str
    strategy: str
    difficulty: str
    needs_tool: bool = False
    needs_json: bool = False
    needs_vision: bool = False
    escalation_chain: list = field(default_factory=list)
    enabled: bool = False                 # False → caller uses plain route_request
    trace: list = field(default_factory=list)

    def route_kwargs(self) -> dict:
        """The kwargs to pass into `route_request` for the single-model pick."""
        return {
            "difficulty": self.difficulty,
            "task_category": self.task_category if self.enabled else None,
            "needs_tool": self.needs_tool,
            "needs_json": self.needs_json,
            "require_vision": self.needs_vision,
        }


def decide(signals: dict | None, *, enabled: bool = True,
           escalation_enabled: bool = False,
           multi_model_enabled: bool = False) -> RoutingDecision:
    """Build a RoutingDecision from the request signals. Fail-open: any error or
    `enabled=False` → a passthrough decision the caller routes via today's
    `route_request` (Property 9)."""
    try:
        return _decide(signals or {}, enabled, escalation_enabled,
                       multi_model_enabled)
    except Exception:  # noqa: BLE001
        return RoutingDecision(task_category="general", strategy=SINGLE,
                               difficulty="standard", enabled=False,
                               trace=[{"error": "meta_router fail-open"}])


def _decide(signals: dict, enabled: bool, escalation_enabled: bool,
            multi_model_enabled: bool) -> RoutingDecision:
    difficulty = str(signals.get("difficulty", "standard")).lower()
    text = signals.get("text", "")
    intent = signals.get("intent")
    category = classify_task(text, intent, difficulty)

    needs_tool = bool(signals.get("needs_tool", False))
    needs_json = bool(signals.get("needs_json", False))
    needs_vision = bool(signals.get("needs_vision", False))

    strat = select_strategy(
        {"difficulty": difficulty, "task_category": category},
        enabled=multi_model_enabled,
    )

    chain = escalation_chain(difficulty) if escalation_enabled else [difficulty]

    return RoutingDecision(
        task_category=category,
        strategy=strat.strategy,
        difficulty=difficulty,
        needs_tool=needs_tool,
        needs_json=needs_json,
        needs_vision=needs_vision,
        escalation_chain=chain,
        enabled=enabled,
        trace=[{"category": category, "strategy": strat.strategy,
                "difficulty": difficulty}],
    )


__all__ = ["RoutingDecision", "decide"]
