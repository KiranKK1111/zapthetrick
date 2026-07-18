"""Offline live-evaluation harness
(live-conversational-intelligence R15; tasks 13.2).

Pins Property 15: deterministic metrics over the live decision fns (no audio /
no keys), baseline compare + regression detection, and dev-only / no runtime
effect.
"""
from __future__ import annotations

from app.eval import live_scenarios
from app.eval.live_baseline import LiveBaselineStore


def test_metrics_shape_and_pass_rate():
    m = live_scenarios.live_metrics()
    assert "overall" in m and "per_category" in m and "false_answer_rate" in m
    assert 0.0 <= m["overall"]["pass_rate"] <= 1.0
    # The committed decision fns should pass their own annotated scenarios.
    assert m["overall"]["pass_rate"] >= 0.95


def test_all_categories_present():
    m = live_scenarios.live_metrics()
    for cat in live_scenarios.CATEGORIES:
        assert cat in m["per_category"]


def test_false_answer_rate_zero_on_current_logic():
    m = live_scenarios.live_metrics()
    # Non-questions must not be answerable → no false answers.
    assert m["false_answer_rate"] == 0.0


def test_baseline_no_baseline_is_report_only(tmp_path):
    store = LiveBaselineStore(path=tmp_path / "nope.json")
    rep = store.compare(live_scenarios.live_metrics())
    assert rep.has_baseline is False
    assert rep.regressed is False


def test_baseline_save_then_no_regression(tmp_path):
    store = LiveBaselineStore(path=tmp_path / "live_baseline.json")
    metrics = live_scenarios.live_metrics()
    store.save(metrics)
    rep = store.compare(metrics)
    assert rep.has_baseline is True
    assert rep.regressed is False
    assert rep.drops == []


def test_baseline_detects_regression(tmp_path):
    store = LiveBaselineStore(path=tmp_path / "live_baseline.json")
    store.save(live_scenarios.live_metrics())
    # Simulate a degraded run: lower overall + raised false-answer rate.
    degraded = {
        "overall": {"total": 19, "passed": 10, "pass_rate": 0.5},
        "per_category": {"question_detection": {"total": 3, "passed": 1, "pass_rate": 0.33}},
        "false_answer_rate": 0.5,
    }
    rep = store.compare(degraded)
    assert rep.regressed is True
    scopes = {d["scope"] for d in rep.drops}
    assert "overall" in scopes
    assert "false_answer_rate" in scopes
