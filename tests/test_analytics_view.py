"""Analytics / audit view (personalization-and-governance R5, task 8.3).

Pins Property 5: aggregation of existing telemetry, read-only, no runtime effect.
"""
from __future__ import annotations

from app.personalization.analytics import summary


def test_summary_shape_and_readonly():
    s = summary()
    assert set(("latency", "routing", "degradation")) <= set(s.keys())
    # Read-only: calling it twice yields the same structure, no side effects.
    s2 = summary()
    assert set(s.keys()) == set(s2.keys())


def test_summary_aggregates_degradation_events():
    from app.quality import degrade
    degrade.reset_events()
    degrade.record_event("retrieval", "boom")
    degrade.record_event("retrieval", "boom2")
    degrade.record_event("memory", "x")
    s = summary()
    by_sub = s["degradation"].get("by_subsystem", {})
    assert by_sub.get("retrieval") == 2 and by_sub.get("memory") == 1


def test_summary_aggregates_latency_observatory():
    from app.perceived.observatory import observatory
    observatory.reset()
    observatory.record_ttft("req-1", 120.0)
    observatory.record_ttft("req-2", 240.0)
    s = summary()
    assert s["latency"]["requests_tracked"] >= 2
    assert s["latency"]["avg_ttft_ms"] == 180.0


def test_summary_never_raises():
    assert isinstance(summary(), dict)
