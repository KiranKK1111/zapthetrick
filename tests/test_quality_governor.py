"""Request governor + pipeline selection (evaluation-and-reliability R5, task 7.3).

Pins Property 6: trivial → fast/early-exit (skips retrieval/planning/validation),
complex → deep, consumes the difficulty label, fail-open to deep.
"""
from __future__ import annotations

from app.quality.governor import Budgets, Pipeline, select_pipeline, FAST, DEEP


def test_trivial_selects_fast_and_early_exits():
    p = select_pipeline("trivial", Budgets())
    assert p.kind == FAST and p.is_fast
    # Early exit: retrieval / planning / validation are skipped (R5.2).
    assert p.skips("retrieval") and p.skips("planning") and p.skips("validation")
    assert "model" in p.stages and "output" in p.stages


def test_hard_and_expert_select_deep():
    assert select_pipeline("hard", Budgets()).kind == DEEP
    p = select_pipeline("expert", Budgets())
    assert p.kind == DEEP
    # Deep pipeline includes the heavy stages (R5.3).
    assert "retrieval" in p.stages and "validation" in p.stages


def test_standard_default_deep_but_fast_under_budget():
    assert select_pipeline("standard", Budgets()).kind == DEEP
    assert select_pipeline("standard", Budgets(quality="fast")).kind == FAST
    assert select_pipeline("standard", Budgets(latency_ms=1000)).kind == FAST


def test_thorough_quality_budget_forces_deep_even_if_trivial():
    assert select_pipeline("trivial", Budgets(quality="thorough")).kind == DEEP


def test_unknown_or_missing_difficulty_is_deep_failopen():
    assert select_pipeline(None, Budgets()).kind == DEEP        # R5.5
    assert select_pipeline("bogus", Budgets()).kind == DEEP


def test_select_never_raises_on_bad_budget():
    # Fail-open: a malformed budget object must not crash selection.
    class _Bad:
        quality = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
    p = select_pipeline("trivial", _Bad())
    assert isinstance(p, Pipeline) and p.kind in (FAST, DEEP)
