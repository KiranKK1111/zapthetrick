"""Phase 4 goal-oriented execution engine — the merged spine's new pieces:
Goal Object (#1), spec validation (#12), the task DAG with parallelism +
checkpoint/resume (#2/#9/#15), the execution ledger (#16), failure preflight
(#18), constraint gate (#7), and acceptance-test wiring (#6).

Deterministic + injectable — no LLM/DB. Pins fail-open + additive behavior.
"""
from __future__ import annotations

import asyncio

from app.orchestration import goal_engine as ge
from app.orchestration.decompose import SubTask
from app.orchestration.state import AgentState


# ── #1 / #12 Goal Object + spec validation ──────────────────────────────────
def test_build_goal_extracts_constraints():
    g = ge.build_goal("Write a parser with tests, under 100 lines, as JSON",
                      condition="parser works")
    assert g.objective.startswith("Write a parser")
    assert g.deliverable == "parser works"
    texts = " ".join(c.text for c in g.constraints)
    assert "tests" in texts and "100" in texts and "JSON" in texts.lower() \
        or "json" in texts.lower()
    assert g.valid


def test_validate_goal_flags_empty():
    g = ge.build_goal("")
    assert not g.valid and g.reasons
    # still usable (fail-open) — acceptance falls back to objective
    assert g.acceptance() == ""


def test_goal_acceptance_prefers_deliverable():
    g = ge.build_goal("do X", deliverable="X is done and verified")
    assert g.acceptance() == "X is done and verified"


# ── #16 execution ledger ────────────────────────────────────────────────────
def test_ledger_records_why():
    led = ge.ExecutionLedger()
    led.record("plan", "decompose", "3 sub-tasks")
    led.record("node", "build API", "deps=none", "ok")
    rows = led.to_list()
    assert len(rows) == 2
    assert rows[0]["kind"] == "plan" and rows[1]["why"] == "deps=none"
    assert rows[0]["step"] == 0 and rows[1]["step"] == 1


# ── #7 constraint gate ──────────────────────────────────────────────────────
def test_constraint_gate_flags_missing_tests():
    g = ge.build_goal("write a function with tests")
    rep = ge.constraint_gate("def f(): return 1", g)
    assert not rep.satisfied
    assert ge.constraint_feedback(rep)


def test_constraint_gate_passes_when_met():
    g = ge.build_goal("write a function with tests")
    rep = ge.constraint_gate("def f(): return 1\n\ndef test_f(): assert f()==1", g)
    assert rep.satisfied and rep.checked >= 1


# ── #18 failure preflight ───────────────────────────────────────────────────
def test_preflight_predicts_offline_network(monkeypatch):
    monkeypatch.setattr(ge, "_network_available", lambda: False)
    rep = ge.preflight("please fetch https://example.com/data and scrape it")
    assert rep is not None and rep.risky
    assert rep.top().failure_id == "network_error"


def test_preflight_clean_task_not_risky():
    rep = ge.preflight("rename a variable in a small file")
    assert rep is None or not rep.risky


# ── #2 / #9 / #15 the task DAG ──────────────────────────────────────────────
def _subs_chain():
    # 0 -> 1 -> 2 (each depends on previous): strict sequential
    return [SubTask(0, "a", []), SubTask(1, "b", [0]), SubTask(2, "c", [1])]


def _subs_diamond():
    # 0 ; 1 and 2 both depend on 0 (parallel level) ; 3 depends on 1,2
    return [SubTask(0, "root", []), SubTask(1, "left", [0]),
            SubTask(2, "right", [0]), SubTask(3, "join", [1, 2])]


def test_dag_runs_in_dependency_order():
    order = []

    async def run_node(sub, prior):
        order.append(sub.id)
        return f"out-{sub.id}"

    res = asyncio.run(ge.execute_dag(_subs_chain(), run_node))
    assert order == [0, 1, 2]
    assert res.outputs == {0: "out-0", 1: "out-1", 2: "out-2"}


def test_dag_parallel_level_and_deps_passed():
    seen_prior = {}

    async def run_node(sub, prior):
        seen_prior[sub.id] = dict(prior)
        await asyncio.sleep(0)
        return f"out-{sub.id}"

    res = asyncio.run(ge.execute_dag(_subs_diamond(), run_node))
    # join saw both upstream outputs
    assert set(seen_prior[3]) == {1, 2}
    # left/right saw root only
    assert set(seen_prior[1]) == {0} and set(seen_prior[2]) == {0}
    assert len(res.order) == 3            # 3 levels: {0},{1,2},{3}
    assert sorted(res.order[1]) == [1, 2]


def test_dag_checkpoint_resume_skips_done():
    ran = []

    async def run_node(sub, prior):
        ran.append(sub.id)
        return f"out-{sub.id}"

    # A state with node 0 already DONE → resume runs only 1,2.
    st = AgentState(goal="dag")
    st.set_tasks(["a", "b", "c"])
    st.mark_done(0, "cached-0")
    saves = []
    res = asyncio.run(ge.execute_dag(
        _subs_chain(), run_node, state=st, save_cb=lambda s: saves.append(1)))
    assert 0 not in ran and 1 in ran and 2 in ran
    assert res.outputs[0] == "cached-0"   # cached output reused
    assert res.resumed and saves           # checkpoint saved after each node


def test_dag_node_failure_is_failopen():
    async def run_node(sub, prior):
        if sub.id == 1:
            raise RuntimeError("boom")
        return f"out-{sub.id}"

    res = asyncio.run(ge.execute_dag(_subs_chain(), run_node))
    assert res.outputs[0] == "out-0"
    assert res.outputs[1] == ""           # failed node → empty, no raise
    assert res.outputs[2] == "out-2"      # run continued


def test_dag_ledger_records_each_node():
    led = ge.ExecutionLedger()

    async def run_node(sub, prior):
        return "x"

    asyncio.run(ge.execute_dag(_subs_chain(), run_node, ledger=led))
    kinds = [e["kind"] for e in led.to_list()]
    assert kinds.count("node") == 3 and "plan" in kinds


# ── #6 acceptance-test engine wiring ────────────────────────────────────────
def test_acceptance_tests_delegates_and_runs():
    gen_calls = []

    async def gen(change):
        gen_calls.append(change)
        return "def test_ok(): assert True"

    async def runner(cmd, workspace_id):
        class _R:
            ok = True
            exit_code = 0
            def summary(self):
                return "1 passed"
        return _R()

    res = asyncio.run(ge.acceptance_tests(
        "ws", "built a thing", gen_fn=gen, test_cmd="pytest",
        runner=runner, force=True))
    assert gen_calls == ["built a thing"]
    assert res.ran and res.passed and res.status == "passed"


# ── #2/#3 RoleRunner wiring ──────────────────────────────────────────────────
def test_plan_role_models_assigns_each_role():
    from app.orchestration.workflow import select_workflow
    wf = select_workflow({"difficulty": "hard", "task_category": "coding"})
    assigned = asyncio.run(ge.plan_role_models(wf, "coding"))
    assert set(assigned) == {"coder", "reviewer", "security_reviewer"}
    assert all(assigned.values())
