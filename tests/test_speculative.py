"""Speculative multi-model drafting (perceived-speed R4, task 6.3).

Pins Properties 4 + 5: race picks the fastest opening, cancels losers, the
winner produces the whole answer (no contradiction), and it falls back to a
single stream when disabled / <2 candidates / all fail.
"""
from __future__ import annotations

import asyncio

from app.core.config_loader import cfg
from app.perceived.budget import SpeculationBudget
from app.perceived.speculative import SpeculativeDrafter


def _enable(monkeypatch):
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", True, raising=False)
    monkeypatch.setattr(cfg.perceived, "speculative_drafting", True, raising=False)
    monkeypatch.setattr(cfg.perceived, "max_concurrent_drafts", 3, raising=False)
    monkeypatch.setattr(cfg.perceived, "speculation_period_budget", 0, raising=False)


def _gen(chunks, *, delay=0.0, consumed=None, tag=""):
    """Build a zero-arg factory → async generator yielding `chunks`."""
    async def _it():
        try:
            for c in chunks:
                if delay:
                    await asyncio.sleep(delay)
                yield c
            if consumed is not None:
                consumed.append(tag)        # only set if fully consumed
        except asyncio.CancelledError:
            raise
    return _it


async def _collect(agen):
    out = []
    async for c in agen:
        out.append(c)
    return out


def test_disabled_falls_back_to_single(monkeypatch):
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", False, raising=False)
    d = SpeculativeDrafter(budget=SpeculationBudget())
    fast = _gen(["A", "B"], delay=0.001)
    slow = _gen(["X", "Y"], delay=0.05)
    out = asyncio.run(_collect(d.race([fast, slow])))
    assert out == ["A", "B"]           # just the first factory, no race


def test_single_candidate_streams_directly(monkeypatch):
    _enable(monkeypatch)
    d = SpeculativeDrafter(budget=SpeculationBudget())
    out = asyncio.run(_collect(d.race([_gen(["only"])])))
    assert out == ["only"]


def test_race_winner_streams_fully_and_losers_cancelled(monkeypatch):
    _enable(monkeypatch)
    b = SpeculationBudget()
    d = SpeculativeDrafter(budget=b)
    consumed = []
    fast = _gen(["F1", "F2", "F3"], delay=0.001, consumed=consumed, tag="fast")
    slow = _gen(["S1", "S2", "S3"], delay=0.05, consumed=consumed, tag="slow")
    out = asyncio.run(_collect(d.race([fast, slow])))
    # Winner (fast) produced the WHOLE answer — no mixing with the loser.
    assert out == ["F1", "F2", "F3"]
    # The slow loser was cancelled, never fully consumed.
    assert "slow" not in consumed


def test_no_opening_falls_back_to_single(monkeypatch):
    _enable(monkeypatch)
    d = SpeculativeDrafter(budget=SpeculationBudget())

    # Candidates that produce NO token (empty streams) → no winner → fall back
    # to a single stream of factory[0] (also empty) → clean, no crash.
    def _empty():
        async def _it():
            if False:        # pragma: no cover
                yield ""
        return _it()

    out = asyncio.run(_collect(d.race([_empty, _empty])))
    assert out == []


def test_draft_error_propagates_like_a_single_stream(monkeypatch):
    _enable(monkeypatch)
    d = SpeculativeDrafter(budget=SpeculationBudget())

    def _boom():
        async def _it():
            raise RuntimeError("draft failed")
            yield  # pragma: no cover
        return _it()

    # All drafts fail → fallback runs factory[0], whose error surfaces to the
    # caller exactly as a normal single stream would (the engine's own fallback
    # chain handles this in production).
    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(_collect(d.race([_boom, _boom])))


def test_concurrency_cap_limits_candidates(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(cfg.perceived, "max_concurrent_drafts", 2, raising=False)
    b = SpeculationBudget()
    d = SpeculativeDrafter(budget=b)
    consumed = []
    # 3 candidates given but cap=2 → the 3rd is never started.
    f1 = _gen(["A"], delay=0.001, consumed=consumed, tag="f1")
    f2 = _gen(["B"], delay=0.05, consumed=consumed, tag="f2")
    f3 = _gen(["C"], delay=0.05, consumed=consumed, tag="f3")
    out = asyncio.run(_collect(d.race([f1, f2, f3])))
    assert out == ["A"]
    assert "f3" not in consumed        # 3rd candidate never ran (cap=2)
