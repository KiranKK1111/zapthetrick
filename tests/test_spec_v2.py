"""Stage-6 §4.2 — speculation v2: trigger gate, flush/miss, enrichment budget."""
from __future__ import annotations

import pytest

from app.live import spec_v2 as SP


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "speculation_v2", True, raising=False)


class TestTrigger:
    def test_plausible_question_fires(self, _on):
        assert SP.should_speculate(
            "What is your experience with Kafka").fire is True

    def test_mid_thought_holds(self, _on):
        # "tell me" — a pronoun after a verb expecting an object → incomplete.
        t = SP.should_speculate("So can you tell me")
        assert t.fire is False and t.completeness == "incomplete"

    def test_too_short_holds(self, _on):
        assert SP.should_speculate("hi there").fire is False

    def test_disabled_never_fires(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "speculation_v2", False, raising=False)
        assert SP.should_speculate("What is your experience with Kafka").fire \
            is False

    def test_never_raises(self, _on):
        SP.should_speculate(None)                     # type: ignore[arg-type]


class TestFlush:
    def test_matching_final_flushes(self):
        d = SP.flush_decision("what is your experience with kafka",
                              "What is your experience with Kafka?")
        assert d.flush is True and d.hedge is False
        assert d.similarity >= 0.92

    def test_divergent_final_hedges(self):
        d = SP.flush_decision(
            "what is kafka",
            "tell me about a time you led a team through conflict")
        assert d.flush is False and d.hedge is True
        assert d.similarity < 0.92

    def test_threshold_boundary(self):
        # Custom threshold: an exact match always flushes.
        d = SP.flush_decision("hello world", "hello world", threshold=0.92)
        assert d.flush is True and d.similarity == 1.0

    def test_empty_final_does_not_flush(self):
        d = SP.flush_decision("what is kafka", "")
        assert d.flush is False and d.hedge is True

    def test_never_raises(self):
        SP.flush_decision(None, None)                 # type: ignore[arg-type]


class TestEnrichmentBudget:
    def test_runs_within_budget(self):
        b = SP.EnrichmentBudget(budget_ms=120)
        assert b.should_run("intent", 50) is True
        assert b.should_run("retrieve", 60) is True   # 110 total ≤ 120
        assert b.spent_ms == 110

    def test_defers_when_over_budget(self):
        b = SP.EnrichmentBudget(budget_ms=120)
        b.should_run("a", 100)
        assert b.should_run("slow", 40) is False       # 140 > 120 → deferred
        assert b.deferred == ["slow"]
        assert b.ran == ["a"]

    def test_free_stage_always_runs(self):
        b = SP.EnrichmentBudget(budget_ms=10)
        b.should_run("a", 10)
        assert b.should_run("free", 0) is True         # 0-cost never deferred

    def test_remaining_ms(self):
        b = SP.EnrichmentBudget(budget_ms=120)
        b.should_run("a", 30)
        assert b.remaining_ms == 90

    def test_default_budget_from_config(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "enrichment_budget_ms", 200.0,
                            raising=False)
        b = SP.EnrichmentBudget()                       # picks up the config
        assert b.budget_ms == 200.0

    def test_as_dict_shape(self):
        b = SP.EnrichmentBudget(budget_ms=120)
        b.should_run("a", 30)
        b.should_run("big", 200)
        d = b.as_dict()
        assert d["ran"] == ["a"] and d["deferred"] == ["big"]
