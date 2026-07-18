"""P2-11 — self-improvement on hard turns (generation council + reflection).

Pure/offline: candidate generation across models (monkeypatched), the
self-consistency vs judge selection, single-draft fallbacks, self_consistency,
the reflect() lesson string, and the loop gating helpers.
"""
from __future__ import annotations

import asyncio

import pytest

from app.chat import self_improve as si


def _patch_drafts(monkeypatch, drafts):
    """Make llm.complete_routed return successive drafts (rotating model ids)."""
    import app.core.llm_client as lc
    seq = iter(list(drafts))
    model_ids = iter([1, 2, 3, 4, 5])

    async def fake_routed(messages, model=None, options=None):
        try:
            return next(seq), next(model_ids, 9)
        except StopIteration:
            return "", None
    monkeypatch.setattr(lc.llm, "complete_routed", fake_routed)


# ── self-consistency selection ───────────────────────────────────────────────
def test_generation_council_consistency_majority(monkeypatch):
    # two identical JSON actions + one different → consistency wins
    a = '{"tool":"read","args":{"path":"x"}}'
    b = '{"tool":"grep","args":{"pattern":"y"}}'
    _patch_drafts(monkeypatch, [a, a, b])
    res = asyncio.run(si.generation_council(
        [{"role": "user", "content": "do it"}], n=3))
    assert res.method == "consistency"
    assert res.n == 3 and res.agreement >= 0.66
    assert res.text == a


def test_generation_council_falls_back_to_judge(monkeypatch):
    # three distinct answers → no majority → judge picks
    _patch_drafts(monkeypatch, ["alpha answer", "beta answer", "gamma answer"])

    async def fake_best_of_n(task, cands):
        return 1, "beta is clearest"
    monkeypatch.setattr("app.chat.council.best_of_n", fake_best_of_n)

    res = asyncio.run(si.generation_council(
        [{"role": "user", "content": "explain"}], n=3))
    assert res.method == "judge"
    assert res.text == "beta answer" and "clearest" in res.why


def test_generation_council_single_draft(monkeypatch):
    _patch_drafts(monkeypatch, ["only one"])
    res = asyncio.run(si.generation_council(
        [{"role": "user", "content": "q"}], n=3))
    assert res.method == "single" and res.text == "only one"


def test_generation_council_total_failure_falls_back_to_complete(monkeypatch):
    import app.core.llm_client as lc

    async def empty_routed(messages, model=None, options=None):
        return "", None
    monkeypatch.setattr(lc.llm, "complete_routed", empty_routed)

    async def fake_complete(messages, model=None, options=None):
        return "plain completion"
    monkeypatch.setattr(lc.llm, "complete", fake_complete)

    res = asyncio.run(si.generation_council(
        [{"role": "user", "content": "q"}], n=3))
    assert res.method == "single" and res.text == "plain completion"


def test_normalize_unifies_json_formatting():
    a = si._normalize('{"tool": "read", "args": {"path": "x"}}')
    b = si._normalize('{"args":{"path":"x"},"tool":"read"}')
    assert a == b   # key order / whitespace ignored


# ── self_consistency ─────────────────────────────────────────────────────────
def test_self_consistency_returns_majority(monkeypatch):
    _patch_drafts(monkeypatch, ["yes", "yes", "no"])
    ans, agreement = asyncio.run(si.self_consistency("is it ok?", n=3))
    assert ans == "yes" and agreement >= 0.66


def test_self_consistency_empty(monkeypatch):
    import app.core.llm_client as lc

    async def empty_routed(messages, model=None, options=None):
        return "", None
    monkeypatch.setattr(lc.llm, "complete_routed", empty_routed)
    ans, agreement = asyncio.run(si.self_consistency("q", n=3))
    assert ans == "" and agreement == 0.0


# ── reflection ────────────────────────────────────────────────────────────────
def test_reflect_success_and_failure():
    ok = si.reflect("build the API", success=True, verify_ok=True, rounds=1)
    assert "succeeded" in ok and "build/tests passed" in ok
    bad = si.reflect("fix bug", success=False, verify_ok=False, rounds=3,
                     issues=["null deref", ""])
    assert "did not fully succeed" in bad and "3 repair round" in bad
    assert "null deref" in bad


# ── loop gating helpers ──────────────────────────────────────────────────────
def test_loop_gating_helpers(monkeypatch):
    from app.agent import loop
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "self_improve", False)
    assert loop._gen_council_on() is False
    monkeypatch.setattr(cfg.advanced_rag, "self_improve", True)
    assert loop._gen_council_on() is True
    monkeypatch.setattr(cfg.advanced_rag, "self_improve_n", 4)
    assert loop._gen_council_n() == 4


def test_loop_uses_council_at_step0_when_enabled(monkeypatch):
    """With self_improve on, step 0 routes through generation_council."""
    import app.core.llm_client as lc
    from app.agent import loop
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.advanced_rag, "self_improve", True)
    monkeypatch.setattr(cfg.advanced_rag, "self_improve_n", 2)

    called = {"council": 0}

    async def fake_council(messages, *, n=3, options=None,
                           consistency_threshold=0.5):
        called["council"] += 1
        return si.CouncilResult(
            text='{"tool":"final","args":{"message":"done via council"}}',
            n=n, method="consistency", agreement=1.0)
    monkeypatch.setattr("app.chat.self_improve.generation_council",
                        fake_council)

    async def boom_complete(messages, model=None, options=None):
        raise AssertionError("step 0 should use the council, not complete")
    monkeypatch.setattr(lc.llm, "complete", boom_complete)

    async def run():
        return [e async for e in loop.run_agent(
            "hard task", workspace=".", mode="acceptEdits")]
    events = asyncio.run(run())
    assert called["council"] == 1
    assert any(e["type"] == "council" for e in events)
    assert any(e["type"] == "final" for e in events)
