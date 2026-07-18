"""Stream analytics — TTFMU / first-code / artifact-ready (P6 #21)."""
from __future__ import annotations

from app.response_arch.analytics import (
    ARTIFACT_READY, DONE, FIRST_CODE, FIRST_MEANINGFUL, StreamAnalytics)


def test_derived_durations():
    clock = {"t": 100.0}
    a = StreamAnalytics(clock=lambda: clock["t"]).start()
    clock["t"] = 100.25
    a.mark(FIRST_MEANINGFUL)
    clock["t"] = 100.5
    a.mark(FIRST_CODE)
    clock["t"] = 100.75
    a.mark(ARTIFACT_READY)
    clock["t"] = 102.0
    a.mark(DONE)
    m = a.metrics()
    assert m["ttfmu_ms"] == 250
    assert m["first_code_ms"] == 500
    assert m["artifact_ready_ms"] == 750
    assert m["total_ms"] == 2000


def test_absent_marks_omitted():
    a = StreamAnalytics().start()
    a.mark(FIRST_MEANINGFUL)
    m = a.metrics()
    assert "ttfmu_ms" in m
    assert "first_code_ms" not in m
    assert "artifact_ready_ms" not in m


def test_mark_is_idempotent_first_wins():
    clock = {"t": 0.0}
    a = StreamAnalytics(clock=lambda: clock["t"]).start()
    clock["t"] = 0.1
    a.mark(FIRST_MEANINGFUL)
    clock["t"] = 5.0
    a.mark(FIRST_MEANINGFUL)          # ignored
    assert a.metrics()["ttfmu_ms"] == 100


def test_never_raises():
    a = StreamAnalytics()
    a.mark(FIRST_MEANINGFUL)          # start() auto-called
    assert isinstance(a.metrics(), dict)
