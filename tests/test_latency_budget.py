"""Latency budget manager (roadmap Phase 5 #1).

Pins the real-budget behaviour: an upstream stage that overruns squeezes the
downstream stages' deadlines so the whole-turn total is honoured — as opposed to
the old static per-stage table where every stage kept its full slice.
"""
from __future__ import annotations

from app.blackboard.budget_manager import LatencyBudget, default_budget


def _clock():
    box = {"t": 0.0}
    return box, (lambda: box["t"])


def test_static_deadline_when_within_budget():
    box, now = _clock()
    b = LatencyBudget(total_ms=8000, stage_deadlines_ms={"intent": 250, "plan": 200},
                      now=now)
    # Nothing elapsed → each stage gets its full static allowance.
    assert b.deadline_for("intent") == 250.0
    assert b.deadline_for("plan") == 200.0


def test_upstream_overrun_squeezes_downstream():
    box, now = _clock()
    b = LatencyBudget(total_ms=1000,
                      stage_deadlines_ms={"a": 500, "b": 500}, now=now)
    assert b.deadline_for("a") == 500.0
    box["t"] = 800.0                       # stage a burned 800ms of the 1000 total
    # Only 200ms of budget remains, so b's 500ms static deadline is clamped.
    assert b.deadline_for("b") == 200.0
    assert b.remaining_ms() == 200.0


def test_over_budget_returns_floor_not_zero():
    box, now = _clock()
    b = LatencyBudget(total_ms=500, stage_deadlines_ms={"x": 400}, now=now)
    box["t"] = 600.0                       # already over the total
    assert b.over_budget()
    d = b.deadline_for("x", floor_ms=1.0)
    assert d == 1.0                        # tiny non-zero grant, never 0/unbounded


def test_unknown_stage_gets_remaining_budget():
    box, now = _clock()
    b = LatencyBudget(total_ms=1000, stage_deadlines_ms={}, now=now)
    box["t"] = 300.0
    assert b.deadline_for("mystery") == 700.0


def test_stage_context_manager_charges_elapsed():
    box, now = _clock()
    b = LatencyBudget(total_ms=1000, stage_deadlines_ms={"s": 900}, now=now)
    with b.stage("s"):
        box["t"] = 400.0                   # simulate 400ms spent inside
    assert b.consumed("s") == 400.0
    assert b.remaining_ms() == 600.0


def test_default_budget_defaults_and_override():
    b = default_budget()
    assert b.total_ms > 0
    assert "intent" in b.stage_deadlines_ms
    b2 = default_budget(deadlines_ms={"total": 5000, "intent": 100})
    assert b2.total_ms == 5000 and b2.stage_deadlines_ms["intent"] == 100


def test_snapshot_shape():
    b = LatencyBudget(total_ms=100, stage_deadlines_ms={"a": 50})
    b.consume("a", 10)
    snap = b.snapshot()
    assert set(snap) >= {"total_ms", "elapsed_ms", "remaining_ms", "over_budget",
                         "consumed"}
