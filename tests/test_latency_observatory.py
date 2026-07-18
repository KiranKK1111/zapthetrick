"""Latency observatory (perceived-speed R16, task 18.3).

Pins Property 12: all stages + TTFT recorded, and the over-budget stage flagged
as the bottleneck.
"""
from __future__ import annotations

from app.perceived.observatory import STAGES, LatencyObservatory


def test_records_stages_and_ttft():
    obs = LatencyObservatory()
    for s in STAGES:
        obs.record("r1", s, 10.0)
    obs.record_ttft("r1", 420.0)
    rep = obs.report("r1")
    assert set(rep["stages"]) == set(STAGES)
    assert rep["ttft_ms"] == 420.0


def test_bottleneck_flags_over_budget_stage():
    obs = LatencyObservatory()
    obs.record("r1", "routing", 50.0)
    obs.record("r1", "retrieval", 900.0)   # blows its budget
    obs.record("r1", "provider", 300.0)
    budgets = {"routing": 100.0, "retrieval": 200.0, "provider": 800.0}
    assert obs.bottleneck("r1", budgets) == "retrieval"


def test_bottleneck_none_when_all_within_budget():
    obs = LatencyObservatory()
    obs.record("r1", "routing", 50.0)
    obs.record("r1", "provider", 100.0)
    budgets = {"routing": 100.0, "provider": 800.0}
    assert obs.bottleneck("r1", budgets) is None


def test_bottleneck_is_slowest_without_budgets():
    obs = LatencyObservatory()
    obs.record("r1", "routing", 50.0)
    obs.record("r1", "streaming", 700.0)
    assert obs.bottleneck("r1") == "streaming"


def test_unknown_request_reports_none():
    obs = LatencyObservatory()
    assert obs.report("nope") is None


def test_unknown_stage_ignored():
    obs = LatencyObservatory()
    obs.record("r1", "not_a_stage", 999.0)
    obs.record("r1", "routing", 5.0)
    assert set(obs.report("r1")["stages"]) == {"routing"}
