"""Task engine + state persistence (agent-orchestration R6/R7, tasks 9.2/10.2).

Pins Properties 6 & 7: long-horizon progress without redo, small→direct,
persist/resume, scope + bound.
"""
from __future__ import annotations

import asyncio

from app.orchestration.task_engine import run_goal
from app.orchestration.state import (
    AgentState, save_state, load_state, DONE,
)


async def _plan(goal):
    return ["analyze repo", "write migration plan", "generate code"]


def _exec_factory(calls):
    async def _exec(task_text, state):
        calls.append(task_text)
        return f"did: {task_text}"
    return _exec


def test_small_goal_direct():
    calls = []
    res = asyncio.run(run_goal("fix a typo", plan_fn=_plan,
                               exec_fn=_exec_factory(calls)))
    assert res.direct is True
    assert calls == ["fix a typo"]          # no planning


def test_large_goal_runs_all_tasks():
    calls = []
    res = asyncio.run(run_goal(
        "1. analyze repo\n2. write migration plan\n3. generate code",
        plan_fn=_plan, exec_fn=_exec_factory(calls)))
    assert res.direct is False
    assert len(res.completed) == 3
    assert res.state.complete


def test_resume_skips_completed_no_redo():
    # A state with task 0 already DONE → resume only runs the rest (no redo).
    st = AgentState(goal="big goal")
    st.set_tasks(["t0", "t1", "t2"])
    st.mark_done(0, "already done")
    calls = []
    res = asyncio.run(run_goal("big goal", plan_fn=_plan,
                               exec_fn=_exec_factory(calls), state=st))
    assert res.resumed is True
    assert "t0" not in calls               # completed task not redone (R6.2)
    assert "t1" in calls and "t2" in calls


def test_state_persist_and_restore_scoped():
    st = AgentState(goal="g", scope="workspace:p1")
    st.set_tasks(["a", "b"])
    st.mark_done(0, "out-a")
    prefs: dict = {}
    save_state(prefs, st)
    # Another scope isn't returned.
    assert load_state(prefs, "workspace:p2") is None
    restored = load_state(prefs, "workspace:p1")
    assert restored is not None
    assert restored.is_done(0) and not restored.is_done(1)
    assert restored.tasks[0].output == "out-a"


def test_bounded_tasks():
    st = AgentState(goal="g")
    st.set_tasks([f"t{i}" for i in range(200)])
    assert len(st.tasks) <= 64
