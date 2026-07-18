"""Strategy selection + meta-router (intelligent-model-routing R6/R7, tasks 8.2/9.3).

Pins Property 6: strategy selection, role assignment + single fallback, the
meta-router unifies signals and delegates the final pick to route_request, and
disabled → passthrough (no LLM call).
"""
from __future__ import annotations

import asyncio
import inspect

from app.llm import strategy as S
from app.llm import meta_router as M


# ── strategy selection ───────────────────────────────────────────────────────
def test_select_strategy_single_by_default():
    d = S.select_strategy({"difficulty": "standard", "task_category": "general"})
    assert d.strategy == S.SINGLE


def test_select_strategy_plan_exec_validate_for_expert_build():
    d = S.select_strategy({"difficulty": "expert", "task_category": "agentic"})
    assert d.strategy == S.PLAN_EXEC_VALIDATE
    assert d.roles == ("planner", "executor", "validator")


def test_select_strategy_consensus_for_hard_reasoning():
    d = S.select_strategy({"difficulty": "hard", "task_category": "math"})
    assert d.strategy == S.CONSENSUS


def test_select_strategy_disabled_is_single():
    d = S.select_strategy({"difficulty": "expert", "task_category": "agentic"},
                          enabled=False)
    assert d.strategy == S.SINGLE


# ── multi-model runner ───────────────────────────────────────────────────────
def test_multimodel_assigns_roles_and_combines():
    async def route_for(role, category):
        return f"model-for-{role}"

    async def generate(role, model):
        return f"{role}-output"

    runner = S.MultiModelRunner(route_for, generate)
    dec = S.select_strategy({"difficulty": "expert", "task_category": "coding"})
    res = asyncio.run(runner.run(dec, "coding"))
    assert res.strategy == S.PLAN_EXEC_VALIDATE
    assert set(res.roles_used) == {"planner", "executor", "validator"}
    assert res.answer == "validator-output"      # validator is the final


def test_multimodel_degrades_to_single_on_failure():
    calls = {"n": 0}

    async def route_for(role, category):
        return "m"

    async def generate(role, model):
        calls["n"] += 1
        if role != "single":
            raise RuntimeError("role failed")
        return "single-answer"

    runner = S.MultiModelRunner(route_for, generate)
    dec = S.select_strategy({"difficulty": "expert", "task_category": "coding"})
    res = asyncio.run(runner.run(dec, "coding"))
    assert res.degraded_to_single and res.answer == "single-answer"


# ── meta-router ──────────────────────────────────────────────────────────────
def test_decide_is_synchronous_no_llm():
    assert not inspect.iscoroutinefunction(M.decide)


def test_decide_classifies_and_builds_route_kwargs():
    dec = M.decide({"text": "write a python function", "difficulty": "standard"},
                   enabled=True)
    assert dec.task_category == "coding"
    kw = dec.route_kwargs()
    assert kw["task_category"] == "coding" and kw["difficulty"] == "standard"


def test_decide_disabled_passthrough_drops_task_category():
    dec = M.decide({"text": "write code", "difficulty": "standard"}, enabled=False)
    # Disabled → route_kwargs must not impose a task_category (today's routing).
    assert dec.enabled is False
    assert dec.route_kwargs()["task_category"] is None


def test_decide_escalation_chain_only_when_enabled():
    on = M.decide({"text": "x", "difficulty": "standard"}, enabled=True,
                  escalation_enabled=True)
    off = M.decide({"text": "x", "difficulty": "standard"}, enabled=True,
                   escalation_enabled=False)
    assert on.escalation_chain == ["standard", "hard"]
    assert off.escalation_chain == ["standard"]


def test_decide_failopen_on_garbage():
    dec = M.decide(None, enabled=True)
    # Empty signals → a safe general/single decision, never a crash.
    assert dec.strategy == S.SINGLE and dec.task_category == "general"
