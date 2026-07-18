"""Multi-agent workflows (agent-orchestration R3, task 4.3).

Pins Property 3: workflow selection, role assignment via routing, verified_answer
reuse for review roles, output combination, and single-agent fallback.
"""
from __future__ import annotations

import asyncio

from app.orchestration import workflow as W


def test_select_single_for_simple():
    wf = W.select_workflow({"difficulty": "standard", "task_category": "general"})
    assert wf.kind == W.SINGLE


def test_select_code_review_for_hard_coding():
    wf = W.select_workflow({"difficulty": "hard", "task_category": "coding"})
    assert wf.kind == W.CODE_REVIEW
    assert wf.roles == ("coder", "reviewer", "security_reviewer")


def test_select_research_workflow():
    wf = W.select_workflow({"difficulty": "expert", "task_category": "research"})
    assert wf.kind == W.RESEARCH


def test_disabled_is_single():
    wf = W.select_workflow({"difficulty": "expert", "task_category": "coding"},
                           enabled=False)
    assert wf.kind == W.SINGLE


def test_role_runner_assigns_models_and_verifies():
    verified_roles = []

    async def route_for(role, category):
        return f"model-{role}"

    async def run_role(role, model, prior):
        return f"{role}:{model}"

    async def verify(role, text):
        verified_roles.append(role)
        return text + "+verified"

    runner = W.RoleRunner(route_for, run_role, verify=verify)
    wf = W.select_workflow({"difficulty": "hard", "task_category": "coding"})
    res = asyncio.run(runner.run(wf, "coding"))
    assert res.kind == W.CODE_REVIEW
    assert set(res.roles_used) == {"coder", "reviewer", "security_reviewer"}
    # Review roles went through verified_answer.
    assert "reviewer" in verified_roles and "security_reviewer" in verified_roles
    assert "verified" in res.answer       # combined output is a verified role's


def test_role_runner_degrades_to_single_when_all_fail():
    async def route_for(role, category):
        return "m"

    async def run_role(role, model, prior):
        if role != "single":
            raise RuntimeError("role failed")
        return "single-answer"

    runner = W.RoleRunner(route_for, run_role)
    wf = W.select_workflow({"difficulty": "expert", "task_category": "agentic"})
    res = asyncio.run(runner.run(wf, "agentic"))
    assert res.degraded_to_single and res.answer == "single-answer"
