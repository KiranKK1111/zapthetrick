"""Adaptive benchmarking (intelligent-model-routing R9, task 12.2).

Pins Property 8: a degraded recent window down-ranks a model; it recovers as the
window fills with healthy samples; an empty window has no opinion.
"""
from __future__ import annotations

from app.llm import adaptive


def setup_function(_):
    adaptive.reset()


def test_empty_window_is_healthy_no_downrank():
    assert adaptive.health_factor("m") == 1.0
    assert adaptive.downrank("m") == 0.0


def test_failures_downrank():
    for _ in range(adaptive._WINDOW):
        adaptive.record_outcome("bad", success=False, latency_ms=12000)
    assert adaptive.health_factor("bad") < 0.5
    assert adaptive.downrank("bad") > 0.0


def test_healthy_window_minimal_downrank():
    for _ in range(adaptive._WINDOW):
        adaptive.record_outcome("good", success=True, latency_ms=500)
    assert adaptive.health_factor("good") > 0.9
    assert adaptive.downrank("good") < 5.0


def test_recovery_as_window_clears():
    # Start degraded...
    for _ in range(adaptive._WINDOW):
        adaptive.record_outcome("m", success=False, latency_ms=12000)
    degraded = adaptive.downrank("m")
    # ...then a run of healthy outcomes pushes the bad ones out of the window.
    for _ in range(adaptive._WINDOW):
        adaptive.record_outcome("m", success=True, latency_ms=400)
    assert adaptive.downrank("m") < degraded
    assert adaptive.health_factor("m") > 0.9


def test_slow_latency_lowers_health_even_when_successful():
    for _ in range(adaptive._WINDOW):
        adaptive.record_outcome("slow", success=True, latency_ms=60000)
    fast_h = 1.0
    assert adaptive.health_factor("slow") < fast_h
