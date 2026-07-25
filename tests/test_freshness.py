"""Tests for the freshness layer (vNext §9.8, Stage 9 Component C)."""
from __future__ import annotations

import app.understanding.freshness as F


def _on(monkeypatch):
    monkeypatch.setattr(F, "enabled", lambda: True)


# ---- classifier -----------------------------------------------------------
def test_semantic_classify_wins(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness("bitcoin value", classify_fn=lambda q: F.VOLATILE)
    assert r.tier == F.VOLATILE and r.source == "semantic"


def test_semantic_out_of_taxonomy_falls_through(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness("stock price right now", classify_fn=lambda q: "banana")
    assert r.tier == F.VOLATILE and r.source == "fallback"


def test_fallback_volatile(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness("what is the weather today", classify_fn=lambda q: None)
    assert r.tier == F.VOLATILE and r.source == "fallback"


def test_fallback_slow(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness("current best practice for packaging",
                             classify_fn=lambda q: None)
    assert r.tier == F.SLOW


def test_slow_beats_volatile_on_latest_version(monkeypatch):
    _on(monkeypatch)
    # "latest version" is SLOW even though it contains the volatile word "latest".
    r = F.classify_freshness("latest version of react", classify_fn=lambda q: None)
    assert r.tier == F.SLOW


def test_stable_default(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness("what is a hash map", classify_fn=lambda q: None)
    assert r.tier == F.STABLE and r.source == "default"


def test_disabled_is_stable(monkeypatch):
    monkeypatch.setattr(F, "enabled", lambda: False)
    r = F.classify_freshness("bitcoin price right now")
    assert r.tier == F.STABLE and r.source == "default"


def test_empty_query_stable(monkeypatch):
    _on(monkeypatch)
    assert F.classify_freshness("", classify_fn=lambda q: None).tier == F.STABLE


def test_classify_never_raises(monkeypatch):
    _on(monkeypatch)
    r = F.classify_freshness(None, classify_fn=lambda q: 1 / 0)  # type: ignore[arg-type]
    assert r.tier == F.STABLE


# ---- strategy -------------------------------------------------------------
def test_stable_strategy_is_direct_no_search():
    s = F.strategy_for(F.STABLE)
    assert s.mode == F.DIRECT and s.needs_search is False


def test_slow_strategy_is_verify():
    s = F.strategy_for(F.SLOW)
    assert s.mode == F.VERIFY and s.needs_search is True
    assert "verification" in s.directive


def test_volatile_strategy_is_search_first():
    s = F.strategy_for(F.VOLATILE)
    assert s.mode == F.SEARCH_FIRST and s.needs_search is True
    assert "search" in s.directive.lower() and "cite" in s.directive.lower()


def test_strategy_accepts_read_object():
    s = F.strategy_for(F.FreshnessRead(F.VOLATILE, 0.8, "semantic"))
    assert s.mode == F.SEARCH_FIRST


def test_strategy_never_raises():
    assert F.strategy_for(None).mode == F.DIRECT   # type: ignore[arg-type]


# ---- live directive -------------------------------------------------------
def test_live_volatile_answers_now_with_verifying():
    d = F.live_directive(F.VOLATILE)
    assert "do NOT wait" in d and "correction" in d


def test_live_slow_answers_now():
    assert "Answer now" in F.live_directive(F.SLOW)


def test_live_stable_is_blank():
    assert F.live_directive(F.STABLE) == ""


def test_live_directive_accepts_read():
    assert F.live_directive(F.FreshnessRead(F.VOLATILE)) != ""


# ---- end-to-end -----------------------------------------------------------
def test_volatile_question_triggers_search_first(monkeypatch):
    _on(monkeypatch)
    read = F.classify_freshness("what is bitcoin worth today", classify_fn=lambda q: None)
    strat = F.strategy_for(read)
    assert read.tier == F.VOLATILE and strat.needs_search and strat.mode == F.SEARCH_FIRST
