"""Tests for the two built-out Phase 3 gaps:
  #16 Temporal / Multi-Horizon   (app/core/temporal.py, wired into TurnState)
  #17 Knowledge Freshness        (app/rag/freshness.py)
Deterministic + offline.
"""
from __future__ import annotations

import pytest

from app.core import temporal as T
from app.core.world_state import TurnState
from app.rag import freshness as F

_DAY = 86_400.0


# ── #16 temporal ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,horizon", [
    ("just tell me quick, what's a mutex?", T.IMMEDIATE),
    ("explain how a hashmap works", T.CONVERSATION),
    ("build me the whole app end to end", T.PROJECT),
    ("help me plan my career over the next year", T.LONG_TERM),
    ("", T.CONVERSATION),
])
def test_classify_horizon(text, horizon):
    assert T.classify_horizon(text) == horizon


def test_horizon_ordering():
    assert T.horizon_ordinal(T.IMMEDIATE) < T.horizon_ordinal(T.PROJECT)
    assert T.horizon_ordinal(T.PROJECT) < T.horizon_ordinal(T.LONG_TERM)


def test_relative_time_and_deadline():
    refs = T.relative_time_refs("I deployed it yesterday and again this morning")
    assert "yesterday" in refs and "this morning" in refs
    assert T.has_deadline("I need this by tomorrow")
    assert not T.has_deadline("explain recursion")


def test_temporal_signal_shape():
    sig = T.temporal_signal("finish the whole project by end of day")
    assert sig["horizon"] == T.PROJECT
    assert sig["deadline"] is True
    assert isinstance(sig["time_refs"], list)


def test_wired_into_turnstate_from_live_snapshot():
    ts = TurnState.from_live_snapshot({"active_question": "build me the entire system"},
                                      goal="build me the entire system")
    assert ts.horizon == T.PROJECT
    assert "horizon" in ts.as_dict()


def test_wired_into_turnstate_from_assessment():
    class _A:  # minimal assessment stand-in
        intent = "chat"; decision = "answer"; confidence = 0.8; ambiguity = 0.1
        risk = 0.0; risk_level = "low"; missing_required = []; matrix = None; policy = None
    ts = TurnState.from_assessment(_A(), goal="quick question, what's a deadlock?",
                                   capabilities=False)
    assert ts.horizon == T.IMMEDIATE
    assert ts.as_dict()["horizon"] == T.IMMEDIATE


# ── #17 freshness ──────────────────────────────────────────────────────────
def test_freshness_decay():
    assert F.freshness_score(0) == 1.0
    half = F.freshness_score(30 * _DAY, half_life_seconds=30 * _DAY)
    assert abs(half - 0.5) < 0.01
    assert F.freshness_score(365 * _DAY) < F.freshness_score(1 * _DAY)


def test_is_stale():
    assert F.is_stale(200 * _DAY)
    assert not F.is_stale(10 * _DAY)


def test_blend_bounds_and_weight():
    # freshness_weight=0 -> pure relevance
    assert F.blend(0.9, 0.1, freshness_weight=0.0) == 0.9
    # higher weight pulls toward freshness
    assert F.blend(0.9, 0.1, freshness_weight=0.5) < 0.9


def test_rerank_by_freshness_prefers_fresh_when_relevance_ties():
    # Same relevance, different ages -> fresher first.
    items = [("old", 0.8, 300 * _DAY), ("new", 0.8, 1 * _DAY)]
    order = [cid for cid, _ in F.rerank_by_freshness(items, freshness_weight=0.3)]
    assert order == ["new", "old"]


def test_rerank_relevance_still_dominates_at_low_weight():
    # A much more relevant but old item stays ahead at small freshness weight.
    items = [("relevant_old", 0.95, 300 * _DAY), ("fresh_weak", 0.30, 0)]
    order = [cid for cid, _ in F.rerank_by_freshness(items, freshness_weight=0.2)]
    assert order[0] == "relevant_old"
