"""
Answer planning (live-conversational-intelligence R9).

Produces a short ordered Answer_Plan (an outline the answer follows). By default
the plan is folded into the SAME generation call as part of the strategy
directive — **no second blocking LLM call**. A heavier two-step plan->generate
is opt-in via `cfg.live.answer_planning_two_step` (not enabled here).
Deterministic + fail-open.
"""
from __future__ import annotations

from app.live import strategy as _strategy

# Per-strategy ordered outline steps (deterministic; mirror the scaffolds).
_STEPS = {
    _strategy.STAR: ["Situation", "Task", "Action", "Result", "Takeaway"],
    _strategy.DESIGN_SESSION: ["Requirements & scale", "High-level architecture",
                              "Key components & data flow", "Trade-offs & bottlenecks"],
    _strategy.CODING_FLOW: ["Restate problem & constraints", "Approach",
                           "Complexity", "Edge cases"],
    _strategy.DEFINITION: ["Definition", "Concrete example", "Where it's used"],
    _strategy.COMPARISON: ["Key dimensions", "Comparison", "Verdict"],
    _strategy.TRADEOFF: ["Options", "Trade-offs", "When each wins"],
    _strategy.DEBUGGING: ["Reproduce", "Hypothesize", "Isolate", "Fix", "Verify"],
    _strategy.GENERAL: [],
}


def make_plan(question: str, strategy: str) -> list[str]:
    """Return an ordered outline for the chosen strategy. Never raises."""
    try:
        return list(_STEPS.get((strategy or "").lower(), []))
    except Exception:  # noqa: BLE001
        return []


def as_directive(plan: list[str]) -> str:
    """Render a plan as a one-line directive for the answer prompt ("" if empty)."""
    steps = [s for s in (plan or []) if s]
    if not steps:
        return ""
    return "Follow this outline: " + " -> ".join(steps) + "."
