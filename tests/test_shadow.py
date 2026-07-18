"""Shadow execution / A-B promotion (roadmap Phase 7 #7)."""
from __future__ import annotations

from app.eval.shadow import (
    run_default_shadow,
    run_shadow,
    should_promote,
)


def test_run_shadow_scores_both_variants():
    cases = [1, 2, 3]
    res = run_shadow(cases, baseline=lambda c: 0.5, candidate=lambda c: 0.9)
    assert res.baseline_mean == 0.5
    assert res.candidate_mean == 0.9
    assert res.improved_cases == 3
    assert res.regressed_cases == 0


def test_should_promote_gate():
    res = run_shadow([1, 2], baseline=lambda c: 0.8, candidate=lambda c: 0.9)
    assert should_promote(res, min_improvement=0.05) is True
    assert should_promote(res, min_improvement=0.5) is False


def test_should_not_promote_on_regression():
    # candidate better on average but regresses one case
    res = run_shadow([1, 2],
                     baseline=lambda c: 0.5 if c == 1 else 0.5,
                     candidate=lambda c: 0.9 if c == 1 else 0.4)
    assert res.regressed_cases == 1
    assert should_promote(res, allow_regressions=0) is False
    assert should_promote(res, allow_regressions=1) is True


def test_default_shadow_consumer_runs():
    """P7 #7: the reachable consumer — default run promotes a clean improvement."""
    out = run_default_shadow()
    assert out["cases"] >= 1
    assert out["candidate_mean"] >= out["baseline_mean"]
    assert out["promote"] is True
    assert out["regressed_cases"] == 0


def test_default_shadow_is_fail_open():
    def boom(_case):
        raise RuntimeError("scorer down")
    # A crashing scorer scores 0 for that variant — no exception escapes.
    out = run_default_shadow(baseline=boom, candidate=boom)
    assert "promote" in out
