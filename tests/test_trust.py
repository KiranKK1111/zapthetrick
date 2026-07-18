"""Phase 8 — quality & trust: confidence band, provenance, red-team review,
and the chat agent-run wiring that surfaces them.
"""
from __future__ import annotations

import asyncio

from app.chat.redteam import Risk, count_high, risks_to_dicts
from app.chat import redteam as rt
from app.chat.trust import (
    ConfidenceSignals,
    build_provenance,
    confidence_band,
)


# ── confidence band ─────────────────────────────────────────────────────────
def test_confidence_high_when_verified_and_passed():
    r = confidence_band(ConfidenceSignals(
        goal_passed=True, verify_attempted=True, verify_ok=True, rounds=1))
    assert r.band == "high" and r.score >= 0.75
    assert any("passed" in x for x in r.reasons)


def test_confidence_low_when_failing():
    r = confidence_band(ConfidenceSignals(
        goal_passed=False, verify_attempted=True, verify_ok=False,
        rounds=4, had_error=True, high_risks=2))
    assert r.band == "low" and r.score < 0.45


def test_confidence_medium_neutral():
    r = confidence_band(ConfidenceSignals())
    assert r.band in ("medium", "high")  # neutral start ~0.7
    assert r.score <= 1.0


def test_confidence_high_risks_penalize():
    base = confidence_band(ConfidenceSignals(goal_passed=True)).score
    risky = confidence_band(ConfidenceSignals(goal_passed=True, high_risks=2)).score
    assert risky < base


# ── provenance ──────────────────────────────────────────────────────────────
def test_provenance_lists_context_and_changes():
    prov = build_provenance(
        context_files=["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"],
        changed_files=3, verify_summary="python: test PASS")
    assert any("6 project file" in p for p in prov)
    assert any("+1 more" in p for p in prov)
    assert any("Edited 3 file" in p for p in prov)
    assert any("Verification" in p for p in prov)


def test_provenance_empty():
    assert build_provenance() == []


# ── red-team review parsing ─────────────────────────────────────────────────
def test_redteam_parse_and_helpers():
    raw = ('{"risks":[{"severity":"high","area":"security",'
           '"issue":"SQL injection in login","fix":"parameterize"},'
           '{"severity":"low","area":"style","issue":"naming"}]}')
    risks = rt._parse(raw)
    assert len(risks) == 2
    assert risks[0].severity == "high" and risks[0].area == "security"
    assert count_high(risks) == 1
    dicts = risks_to_dicts(risks)
    assert dicts[0]["issue"].startswith("SQL injection")


def test_redteam_parse_garbage_is_empty():
    assert rt._parse("not json") == []
    assert rt._parse('{"risks": "nope"}') == []


def test_redteam_review_empty_work_skips():
    out = asyncio.run(rt.red_team_review("task", ""))
    assert out == []


def test_redteam_review_uses_llm(monkeypatch):
    async def _fake(messages, **kw):
        return '{"risks":[{"severity":"high","area":"correctness","issue":"off-by-one"}]}'
    monkeypatch.setattr(rt.llm, "complete", _fake)
    out = asyncio.run(rt.red_team_review("fix loop", "for i in range(n+1): ..."))
    assert len(out) == 1 and out[0].severity == "high"


def test_redteam_review_llm_failure_graceful(monkeypatch):
    from app.core.llm_client import LLMError

    async def _boom(*a, **k):
        raise LLMError("down")
    monkeypatch.setattr(rt.llm, "complete", _boom)
    assert asyncio.run(rt.red_team_review("t", "some work")) == []


# ── endpoint wiring: review/confidence/provenance frames emitted ────────────
def _collect(resp):
    async def go():
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
        return out
    return asyncio.run(go())


async def _scripted(*_a, **_k):
    yield {"type": "goal_round", "round": 1, "of": 4}
    yield {"type": "tool_call", "tool": "edit", "args": {"path": "x.py"}, "step": 1}
    yield {"type": "tool_result", "tool": "edit", "result": "edited x.py"}
    yield {"type": "goal_eval", "round": 1, "passed": True, "verify": "python: test PASS"}
    yield {"type": "goal_done", "passed": True, "rounds": 1}
    yield {"type": "final", "message": "Implemented the change."}


def test_agent_run_emits_trust_frames(monkeypatch):
    from app.api import routes_chat_agent as rca

    monkeypatch.setattr(rca, "_resolve_kind", lambda b: "edit")
    monkeypatch.setattr(rca, "_resolve_workspace", lambda c, k: ("/tmp/ws", ""))

    async def _diff(_ws):
        return "Changed files:\nM\tx.py\n x.py | 5 +++--"
    monkeypatch.setattr(rca, "_diff", _diff)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: None)
    import app.agent.loop as loop
    monkeypatch.setattr(loop, "run_goal", _scripted)

    # Fake the red-team review so no real LLM call happens.
    async def _review(task, work):
        return [Risk(severity="medium", area="edge-case", issue="empty input")]
    monkeypatch.setattr(rt, "red_team_review", _review)

    resp = asyncio.run(rca.chat_agent_run(rca.ChatAgentRunBody(
        conversation_id="c1", task="fix the bug", kind="edit")))
    joined = "".join(_collect(resp))
    assert "event: review" in joined
    assert "event: confidence" in joined
    assert "event: provenance" in joined
    assert '"band"' in joined
