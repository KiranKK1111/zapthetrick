"""Phase 12 — multi-provider council / cross-model verify (B1/B2).

Judges are LLM calls; mocked so tests are offline/deterministic.
"""
from __future__ import annotations

import asyncio

from app.chat import council as cc
from app.chat.council import CouncilVerdict, best_of_n, cross_model_verify


def _routed(seq):
    """Build a fake llm.complete_routed yielding (text, model_db_id) per call,
    and record the avoid_model_db_id passed each call."""
    calls = []
    it = iter(seq)

    async def _fn(messages, options=None, **kw):
        calls.append((options or {}).get("avoid_model_db_id"))
        return next(it)
    return _fn, calls


# ── cross-model verify (B1, n=1) ────────────────────────────────────────────
def test_cross_verify_single_agree(monkeypatch):
    fn, _ = _routed([('{"agree": true, "issues": []}', 11)])
    monkeypatch.setattr(cc.llm, "complete_routed", fn)
    v = asyncio.run(cross_model_verify("task", "good work", n=1))
    assert v.agree and v.votes == 1 and v.agreement == 1.0
    assert v.verifiers == ["11"]


def test_cross_verify_single_disagree(monkeypatch):
    fn, _ = _routed([('{"agree": false, "issues": ["off-by-one", "no tests"]}', 7)])
    monkeypatch.setattr(cc.llm, "complete_routed", fn)
    v = asyncio.run(cross_model_verify("task", "buggy", n=1))
    assert not v.agree and v.votes == 1
    assert "off-by-one" in v.issues


# ── council vote (B2, n=3) ──────────────────────────────────────────────────
def test_council_majority_and_avoids_models(monkeypatch):
    fn, calls = _routed([
        ('{"agree": true, "issues": []}', 1),
        ('{"agree": false, "issues": ["edge case"]}', 2),
        ('{"agree": true, "issues": []}', 3),
    ])
    monkeypatch.setattr(cc.llm, "complete_routed", fn)
    v = asyncio.run(cross_model_verify("task", "work", n=3))
    assert v.votes == 3
    assert v.agree                      # 2 of 3 agree → majority
    assert round(v.agreement, 2) == 0.67
    assert v.verifiers == ["1", "2", "3"]
    assert "edge case" in v.issues
    # Each judge after the first avoids the previous judge's model.
    assert calls == [None, 1, 2]


def test_council_minority_agree_fails(monkeypatch):
    fn, _ = _routed([
        ('{"agree": false, "issues": ["x"]}', 1),
        ('{"agree": false, "issues": ["y"]}', 2),
        ('{"agree": true, "issues": []}', 3),
    ])
    monkeypatch.setattr(cc.llm, "complete_routed", fn)
    v = asyncio.run(cross_model_verify("task", "work", n=3))
    assert not v.agree and round(v.agreement, 2) == 0.33


def test_cross_verify_empty_work_is_neutral():
    v = asyncio.run(cross_model_verify("task", "   ", n=3))
    assert v.agree and v.votes == 0


def test_cross_verify_all_judges_fail_is_neutral(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(cc.llm, "complete_routed", _boom)
    v = asyncio.run(cross_model_verify("task", "work", n=2))
    assert v.agree and v.votes == 0     # nobody voted → don't penalize


def test_cross_verify_unparseable_skipped(monkeypatch):
    fn, _ = _routed([("not json", 1), ('{"agree": true}', 2)])
    monkeypatch.setattr(cc.llm, "complete_routed", fn)
    v = asyncio.run(cross_model_verify("task", "work", n=2))
    assert v.votes == 1 and v.agree     # only the parseable vote counts


# ── best-of-N selection ─────────────────────────────────────────────────────
def test_best_of_n_picks_judged_winner(monkeypatch):
    async def _complete(messages, options=None, **kw):
        return '{"best": 2, "why": "more complete"}'
    monkeypatch.setattr(cc.llm, "complete", _complete)
    idx, why = asyncio.run(best_of_n("task", ["A", "B", "C"]))
    assert idx == 1 and "complete" in why


def test_best_of_n_single_candidate_shortcuts():
    idx, why = asyncio.run(best_of_n("task", ["only"]))
    assert idx == 0


def test_best_of_n_out_of_range_falls_back(monkeypatch):
    async def _complete(messages, options=None, **kw):
        return '{"best": 9}'
    monkeypatch.setattr(cc.llm, "complete", _complete)
    idx, _ = asyncio.run(best_of_n("task", ["A", "B"]))
    assert idx == 0


def test_council_verdict_to_dict():
    d = CouncilVerdict(agree=False, agreement=0.5, votes=2,
                       verifiers=["1", "2"], issues=["x"]).to_dict()
    assert d["agree"] is False and d["votes"] == 2 and d["issues"] == ["x"]


# ── endpoint wiring: cross_verify frame + confidence penalty ────────────────
def _collect(resp):
    async def go():
        return [c if isinstance(c, str) else c.decode()
                async for c in resp.body_iterator]
    return asyncio.run(go())


async def _scripted(*_a, **_k):
    yield {"type": "goal_done", "passed": True, "rounds": 1}
    yield {"type": "final", "message": "implemented"}


def test_agent_run_emits_cross_verify(monkeypatch):
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

    # A council that disagrees → expect a cross_verify frame + low/medium conf.
    async def _verify(task, work, n=1, avoid_model_db_id=None):
        return CouncilVerdict(agree=False, agreement=0.0, votes=2,
                              verifiers=["1", "2"], issues=["broken"])
    monkeypatch.setattr(cc, "cross_model_verify", _verify)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="c1", task="fix it", kind="edit")))
    joined = "".join(_collect(resp))
    assert "event: cross_verify" in joined
    assert '"agree": false' in joined
    assert '"broken"' in joined
