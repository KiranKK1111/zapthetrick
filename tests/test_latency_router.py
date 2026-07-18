"""Provider health + latency-aware ranking (perceived-speed R9, task 11.3).

Pins Property 7: monotonic difficulty selection (simple→fast, complex→strong),
health down-rank, and unavailable models pushed last.
"""
from __future__ import annotations

from app.perceived.health import LatencyRouter, ProviderHealth


def _cands():
    # lower rank = stronger / faster
    return [
        {"id": "fast", "intelligence_rank": 40, "speed_rank": 5},
        {"id": "strong", "intelligence_rank": 5, "speed_rank": 40},
    ]


def test_trivial_prefers_fast():
    r = LatencyRouter(ProviderHealth())
    order = r.select("trivial", _cands())
    assert order[0]["id"] == "fast"


def test_expert_prefers_strong():
    r = LatencyRouter(ProviderHealth())
    order = r.select("expert", _cands())
    assert order[0]["id"] == "strong"


def test_health_downranks_slow_and_failing():
    h = ProviderHealth()
    # "strong" is slow + erroring; "fast" is healthy.
    for _ in range(10):
        h.record("strong", latency_s=8.0, ok=False)
        h.record("fast", latency_s=0.2, ok=True)
    r = LatencyRouter(h)
    # Even on an expert turn (which prefers strong), a badly-degraded strong
    # model is down-ranked below the healthy one.
    order = r.select("expert", _cands())
    assert order[0]["id"] == "fast"


def test_unavailable_pushed_last():
    h = ProviderHealth()
    for _ in range(10):
        h.record("strong", ok=False)   # error_rate ~1.0 → unavailable
    r = LatencyRouter(h)
    order = r.select("expert", _cands())
    assert order[-1]["id"] == "strong"


def test_neutral_with_no_samples():
    h = ProviderHealth()
    assert h.health_score("anything") == 1.0
    assert h.available("anything") is True
    assert h.latency_p50("anything") is None
