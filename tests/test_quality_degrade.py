"""Graceful degradation guard (evaluation-and-reliability R6, task 9.3).

Pins Property 7: a non-critical failure yields a safe fallback + a recorded
event; success passes through; the guard never wraps/bypasses a protected
safety subsystem.
"""
from __future__ import annotations

import asyncio

from app.quality import degrade


def setup_function(_):
    degrade.reset_events()


def test_success_passes_through_no_event():
    out = degrade.guard(lambda: 42, fallback=-1, name="retrieval")
    assert out == 42
    assert degrade.recent_events() == []


def test_failure_returns_fallback_and_records_event():
    out = degrade.guard(lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                        fallback="safe", name="retrieval")
    assert out == "safe"
    events = degrade.recent_events()
    assert len(events) == 1 and events[0]["subsystem"] == "retrieval"


def test_async_guard():
    async def boom():
        raise ValueError("nope")

    out = asyncio.run(degrade.guard_async(boom, fallback=[], name="memory"))
    assert out == []
    assert degrade.recent_events()[0]["subsystem"] == "memory"


def test_protected_safety_not_guarded():
    assert degrade.is_protected("safety") is True
    assert degrade.is_protected("destructive_action") is True
    assert degrade.is_protected("retrieval") is False

    # safe_guard re-raises for a protected subsystem (never a silent fallback).
    import pytest
    with pytest.raises(RuntimeError):
        degrade.safe_guard(
            lambda: (_ for _ in ()).throw(RuntimeError("safety must surface")),
            fallback="swallowed", name="safety")
    # No degradation event recorded for the protected path.
    assert degrade.recent_events() == []


def test_snapshot_and_since_scope_per_turn():
    s0 = degrade.snapshot()
    degrade.guard(lambda: (_ for _ in ()).throw(RuntimeError("x")),
                  fallback=None, name="reranking")
    new = degrade.since(s0)
    assert [e["subsystem"] for e in new] == ["reranking"]
