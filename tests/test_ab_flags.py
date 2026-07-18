"""A/B flag eval comparison (gap G11)."""
from __future__ import annotations

from eval.ab_flags import compare_reports


def _report(**cats):
    return {"summary": {"per_category": {
        c: {"passed": p, "total": t, "avg_score": s}
        for c, (p, t, s) in cats.items()}}}


def test_improvement_verdict():
    off = _report(coding=(6, 10, 0.6), writing=(8, 10, 0.8))
    on = _report(coding=(9, 10, 0.85), writing=(8, 10, 0.8))
    r = compare_reports(off, on)
    assert r["verdict"] == "improved"
    assert r["overall_pass_rate_delta"] > 0
    assert "coding" in r["improved_categories"]


def test_regression_verdict():
    off = _report(coding=(9, 10, 0.9))
    on = _report(coding=(5, 10, 0.5))
    r = compare_reports(off, on)
    assert r["verdict"] == "regressed"
    assert "coding" in r["regressed_categories"]


def test_neutral_verdict():
    off = _report(coding=(8, 10, 0.8))
    on = _report(coding=(8, 10, 0.81))
    assert compare_reports(off, on)["verdict"] == "neutral"


def test_mixed_with_a_regression_is_regressed():
    off = _report(coding=(5, 10, 0.5), writing=(9, 10, 0.9))
    on = _report(coding=(9, 10, 0.9), writing=(6, 10, 0.6))   # writing regressed
    r = compare_reports(off, on)
    assert r["verdict"] == "regressed"
    assert "writing" in r["regressed_categories"]
    assert "coding" in r["improved_categories"]


def test_per_category_deltas():
    off = _report(coding=(5, 10, 0.5))
    on = _report(coding=(8, 10, 0.7))
    d = compare_reports(off, on)["per_category"]["coding"]
    assert abs(d["pass_rate_delta"] - 0.3) < 1e-6
    assert abs(d["avg_score_delta"] - 0.2) < 1e-6


def test_fail_open_on_garbage():
    assert compare_reports({}, {})["verdict"] == "neutral"
    assert "verdict" in compare_reports(None, None)
