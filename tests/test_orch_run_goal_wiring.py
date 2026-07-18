"""Phase 4 wiring — the goal spine (`app/agent/loop.run_goal`) actually executes
the task DAG (#2), emits the execution ledger (#16), and drives the merged
Phase-4 pieces. run_agent + verification are faked so this is offline.
"""
from __future__ import annotations

import asyncio
import tempfile

import app.agent.loop as loop


def _collect(agen):
    async def go():
        out = []
        async for e in agen:
            out.append(e)
        return out
    return asyncio.run(go())


async def _fake_run_agent(prompt, *, workspace=None, mode=None, max_steps=24,
                          context="", history=None, images=None,
                          session_key=None, avoid_model_db_id=None, _depth=0):
    # The evaluator sub-agent asks for a JSON verdict; everyone else gets a
    # short 'done' final. A JSON verdict makes _evaluate pass.
    yield {"type": "thought", "text": "working", "step": 0}
    yield {"type": "final", "message": '{"passed": true, "feedback": ""}'}


def test_run_goal_executes_dag_and_emits_ledger(monkeypatch):
    monkeypatch.setattr(loop, "run_agent", _fake_run_agent)
    # verification is best-effort; force it to "nothing attempted".
    import app.agent_workspace.verify as vmod
    async def _vw(*a, **k):
        return None
    monkeypatch.setattr(vmod, "verify_workspace", _vw)

    task = "1. build the API\n2. write the tests\n3. add the docs"
    with tempfile.TemporaryDirectory() as ws:
        events = _collect(loop.run_goal(
            task, "the app is built", workspace=ws, max_rounds=1))

    types = [e["type"] for e in events]
    assert "graph_start" in types           # the DAG actually ran
    assert types.count("graph_node") == 3    # one per sub-task
    assert "graph_done" in types
    assert "ledger" in types                 # execution ledger emitted
    assert "goal_done" in types
    done = [e for e in events if e["type"] == "goal_done"][0]
    assert done["passed"] is True
    # ledger recorded the plan + each node
    led = [e for e in events if e["type"] == "ledger"][0]["entries"]
    kinds = [row["kind"] for row in led]
    assert kinds.count("node") == 3 and "plan" in kinds


def test_run_goal_single_goal_keeps_classic_loop(monkeypatch):
    monkeypatch.setattr(loop, "run_agent", _fake_run_agent)
    import app.agent_workspace.verify as vmod
    async def _vw(*a, **k):
        return None
    monkeypatch.setattr(vmod, "verify_workspace", _vw)

    with tempfile.TemporaryDirectory() as ws:
        events = _collect(loop.run_goal(
            "fix a typo in the readme", "typo fixed", workspace=ws,
            max_rounds=1))
    types = [e["type"] for e in events]
    # a single simple goal does NOT trigger the DAG path
    assert "graph_start" not in types
    assert "goal_done" in types
