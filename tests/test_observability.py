"""Phase 9 — observability (per-run metrics) + eval harness.

Pure/offline: metric accumulation, ledger aggregation, the offline eval suite,
and the chat agent-run `metrics` SSE frame.
"""
from __future__ import annotations

import asyncio

from app.eval.harness import EvalCase, default_suite, run_suite
from app.obs.metrics import RunMetrics, aggregate_runs, est_tokens


# ── RunMetrics accumulation ─────────────────────────────────────────────────
def test_metrics_accumulate_from_events():
    m = RunMetrics(kind="edit")
    for evt in [
        {"type": "goal_round", "round": 1, "of": 4},
        {"type": "tool_call", "tool": "read", "args": {}},
        {"type": "tool_result", "tool": "read", "result": "x" * 40},
        {"type": "tool_call", "tool": "edit", "args": {}},
        {"type": "tool_result", "tool": "edit", "result": "ok"},
        {"type": "goal_eval", "round": 1, "passed": True, "verify": "test PASS"},
        {"type": "goal_done", "passed": True, "rounds": 2},
        {"type": "final", "message": "done " * 10},
    ]:
        m.on_event(evt)
    m.finalize(confidence="high")
    d = m.to_dict()
    assert d["tool_calls"] == 2
    assert d["rounds"] == 2
    assert d["verify_ok"] is True
    assert d["goal_passed"] is True
    assert d["success"] is True
    assert d["confidence"] == "high"
    assert d["out_tokens"] > 0
    assert d["duration_ms"] >= 0


def test_metrics_counts_errors():
    m = RunMetrics()
    m.on_event({"type": "error", "detail": "boom"})
    m.finalize()
    assert m.to_dict()["errors"] == 1


def test_est_tokens():
    assert est_tokens("") == 0
    assert est_tokens("a" * 40) == 10


# ── ledger aggregation ──────────────────────────────────────────────────────
def test_aggregate_runs():
    runs = [
        {"status": "ok", "tokens": 100,
         "output_summary": {"duration_ms": 2000, "tool_calls": 3}},
        {"status": "error", "tokens": 50,
         "output_summary": {"duration_ms": 1000, "tool_calls": 1}},
    ]
    agg = aggregate_runs(runs)
    assert agg["runs"] == 2
    assert agg["successes"] == 1
    assert agg["success_rate"] == 0.5
    assert agg["total_tokens"] == 150
    assert agg["total_duration_ms"] == 3000
    assert agg["avg_duration_ms"] == 1500
    assert agg["total_tool_calls"] == 4


def test_aggregate_empty():
    assert aggregate_runs([]) == {"runs": 0}


# ── eval harness ────────────────────────────────────────────────────────────
def test_run_suite_pass_fail_and_exceptions():
    cases = [
        EvalCase("ok", lambda: 2 + 2, lambda v: v == 4),
        EvalCase("bad", lambda: 1, lambda v: v == 2),
        EvalCase("boom", lambda: (_ for _ in ()).throw(ValueError("x")),
                 lambda v: True),
    ]
    rep = run_suite(cases)
    assert rep.total == 3 and rep.passed == 1 and rep.failed == 2
    assert rep.pass_rate == round(1 / 3, 3)
    boom = [r for r in rep.results if r.name == "boom"][0]
    assert not boom.passed and "raised" in boom.detail


def test_default_suite_runs_offline_and_passes():
    rep = run_suite(default_suite())
    # Every deterministic-component case should pass (regression guard).
    assert rep.total >= 10
    failed = [r.name for r in rep.results if not r.passed]
    assert failed == [], f"unexpected eval failures: {failed}"
    assert rep.pass_rate == 1.0
    d = rep.to_dict()
    assert d["passed"] == d["total"]


# ── endpoint emits a metrics frame ──────────────────────────────────────────
def _collect(resp):
    async def go():
        return [c if isinstance(c, str) else c.decode()
                async for c in resp.body_iterator]
    return asyncio.run(go())


async def _scripted(*_a, **_k):
    yield {"type": "tool_call", "tool": "edit", "args": {"path": "x.py"}}
    yield {"type": "tool_result", "tool": "edit", "result": "edited"}
    yield {"type": "goal_done", "passed": True, "rounds": 1}
    yield {"type": "final", "message": "done"}


def test_agent_run_emits_metrics_frame(monkeypatch):
    from app.api import routes_chat_agent as rca
    from app.chat import redteam as rt

    monkeypatch.setattr(rca, "_resolve_kind", lambda b: "edit")
    monkeypatch.setattr(rca, "_resolve_workspace", lambda c, k: ("/tmp/ws", ""))

    async def _diff(_ws):
        return ""
    monkeypatch.setattr(rca, "_diff", _diff)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: None)
    import app.agent.loop as loop
    monkeypatch.setattr(loop, "run_goal", _scripted)

    async def _review(task, work):
        return []
    monkeypatch.setattr(rt, "red_team_review", _review)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="c1", task="fix it", kind="edit")))
    joined = "".join(_collect(resp))
    assert "event: metrics" in joined
    assert '"tool_calls"' in joined
    assert '"duration_ms"' in joined
