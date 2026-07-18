"""Orchestration invariants (agent-orchestration R8, task 11.3).

Pins Property 8: bounded + fail-open + additive — every entry point is
sync-or-injectable, never raises on garbage, and a full
decompose→workflow→tool-plan pass composes deterministically.
"""
from __future__ import annotations

import asyncio
import inspect

from app.orchestration import decompose, plan_tools, select_workflow
from app.orchestration import workflow as W
from app.orchestration.sandbox import run_code
from app.orchestration.tests_gen import generate_and_run
from app.orchestration.task_engine import run_goal


def test_pure_entrypoints_are_sync():
    for fn in (decompose, plan_tools, select_workflow):
        assert not inspect.iscoroutinefunction(fn)


def test_full_planning_pass_composes():
    req = "1. analyze the repo\n2. write a migration plan\n3. generate the code"
    subs = decompose(req)
    assert len(subs) == 3
    wf = select_workflow({"difficulty": "expert", "task_category": "coding",
                          "multi_goal": True})
    assert wf.kind in (W.CODE_REVIEW, W.PLAN_EXEC_VALIDATE)
    plan = plan_tools(req, subs, [])     # no tools available → empty plan
    assert plan.names() == []


def test_all_failopen_on_garbage():
    assert decompose(None) == []
    assert select_workflow(None).kind == W.SINGLE
    assert plan_tools(None, None, None).names() == []
    # async ones never raise either
    assert asyncio.run(run_code("ws", None, force=True)).verified is False
    assert asyncio.run(generate_and_run("ws", None, force=True)).status in (
        "skipped", "no_tests", "error")


def test_orchestration_flags_default_off():
    from app.core.config_loader import OrchestrationSection
    o = OrchestrationSection()
    assert o.enabled is False
    assert o.sandbox_verify is False and o.generate_tests is False
