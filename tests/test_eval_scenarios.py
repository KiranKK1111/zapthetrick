"""Scenario coverage matrix (evaluation-and-reliability R1, task 1.3).

Pins Properties 1 & 2: every category present, deterministic with no provider
keys, and category_metrics computes per-category + overall + false-ask/misroute
rates.
"""
from __future__ import annotations

from app.eval.harness import run_suite
from app.eval.scenarios import (
    CATEGORIES, scenarios, scenario_suite, category_metrics,
)


def test_every_category_present():
    cats = {s.category for s in scenarios()}
    for c in CATEGORIES:
        assert c in cats, f"missing scenario category: {c}"


def test_matrix_runs_deterministically_no_keys():
    # Two runs (no provider keys) → identical pass/fail per case (Property 1).
    r1 = run_suite(scenario_suite())
    r2 = run_suite(scenario_suite())
    assert [x.passed for x in r1.results] == [x.passed for x in r2.results]
    # The deterministic matrix should be (near-)green.
    assert r1.pass_rate >= 0.9


def test_category_metrics_shape():
    report = run_suite(scenario_suite())
    m = category_metrics(report)
    assert "overall" in m and "per_category" in m
    assert "false_ask_rate" in m and "misroute_rate" in m
    assert 0.0 <= m["overall"]["pass_rate"] <= 1.0
    # Per-category entries carry total/passed/pass_rate.
    for cat, d in m["per_category"].items():
        assert {"total", "passed", "pass_rate"} <= set(d.keys())


def test_false_ask_and_misroute_rates_are_low_on_baseline():
    m = category_metrics(run_suite(scenario_suite()))
    # The matrix is authored so specific requests don't false-ask and builds
    # route correctly.
    assert m["false_ask_rate"] <= 0.0
    assert m["misroute_rate"] <= 0.0
