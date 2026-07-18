"""Baseline + regression gate (evaluation-and-reliability R2/R8, task 2.3).

Pins Property 3: a metric drop beyond tolerance is flagged + reported; an absent
baseline is report-only; the CLI exits non-zero in CI mode on a regression.
"""
from __future__ import annotations

import json

from app.eval.baseline import BaselineStore, run_matrix


def _metrics(overall_rate, cat_rate=1.0, false_ask=0.0, misroute=0.0):
    return {
        "overall": {"total": 10, "passed": int(overall_rate * 10),
                    "pass_rate": overall_rate},
        "per_category": {"follow_up": {"total": 2, "passed": int(cat_rate * 2),
                                       "pass_rate": cat_rate}},
        "false_ask_rate": false_ask,
        "misroute_rate": misroute,
    }


def test_regression_flagged_beyond_tolerance(tmp_path):
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(_metrics(1.0, 1.0))
    rep = store.compare(_metrics(0.8, 1.0), tolerance=0.02)
    assert rep.has_baseline and rep.regressed
    assert any(d["scope"] == "overall" for d in rep.drops)


def test_within_tolerance_not_flagged(tmp_path):
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(_metrics(1.0, 1.0))
    rep = store.compare(_metrics(0.99, 1.0), tolerance=0.02)
    assert rep.has_baseline and not rep.regressed


def test_category_drop_flagged(tmp_path):
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(_metrics(1.0, 1.0))
    rep = store.compare(_metrics(1.0, 0.5), tolerance=0.02)
    assert rep.regressed
    assert any(d["scope"] == "category:follow_up" for d in rep.drops)


def test_error_rate_climb_flagged(tmp_path):
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(_metrics(1.0, 1.0, false_ask=0.0, misroute=0.0))
    rep = store.compare(_metrics(1.0, 1.0, false_ask=0.2, misroute=0.0),
                        tolerance=0.02)
    assert rep.regressed
    assert any(d["scope"] == "false_ask_rate" for d in rep.drops)


def test_absent_baseline_is_report_only(tmp_path):
    store = BaselineStore(tmp_path / "missing.json")
    rep = store.compare(_metrics(0.1), tolerance=0.02)
    assert rep.has_baseline is False and rep.regressed is False


def test_corrupt_baseline_is_report_only(tmp_path):
    p = tmp_path / "baseline.json"
    p.write_text("{ not valid json", encoding="utf-8")
    rep = BaselineStore(p).compare(_metrics(0.1))
    assert rep.has_baseline is False and rep.regressed is False


def test_run_matrix_produces_metrics():
    m = run_matrix()
    assert "overall" in m and 0.0 <= m["overall"]["pass_rate"] <= 1.0
