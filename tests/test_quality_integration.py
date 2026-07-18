"""Reliability invariants (evaluation-and-reliability R9, task 11.3).

Pins Properties 9 & 10 at the unit level:
- single blocking LLM call: confidence aggregation, the governor, the critic and
  the degradation guard are all synchronous, deterministic functions — none is a
  coroutine / model call, so no second blocking call is added;
- safety precedence: the degradation guard refuses to wrap a protected
  safety/destructive-action subsystem (it surfaces, never silently degrades).
"""
from __future__ import annotations

import inspect

from app.quality import confidence as qc
from app.quality import governor as gov
from app.quality import degrade
from app.quality.critic import review


def test_quality_entrypoints_are_synchronous_no_llm():
    for fn in (qc.aggregate, qc.gate, gov.select_pipeline, review,
               degrade.guard, degrade.safe_guard):
        assert not inspect.iscoroutinefunction(fn), f"{fn.__name__} must be sync"


def test_full_quality_pass_runs_without_provider():
    """A complete confidence→governor→critic pass runs with no model/provider."""
    sigs = [qc.from_routing("standard"),
            qc.from_resolution(type("R", (), {"refs": ["it"], "confidence": 0.8,
                                              "needs_clarification": False})())]
    agg = qc.aggregate(sigs)
    decision = qc.gate(agg)
    pipe = gov.select_pipeline("standard", gov.Budgets())
    rep = review("answer text", asked_items=["x"], decisions={"database": "postgres"})
    assert decision in ("proceed", "clarify", "judgment")
    assert pipe.kind in (gov.FAST, gov.DEEP)
    assert rep is not None


def test_low_confidence_defers_to_clarifier_no_new_path():
    low = qc.aggregate([qc.SubsystemConfidence("trust", 0.1, ["failing"])])
    assert qc.gate(low) == "clarify"      # defers to EXISTING clarifier


def test_degradation_never_bypasses_safety():
    # safe_guard must re-raise (surface) for a protected subsystem (R6.4).
    import pytest
    with pytest.raises(RuntimeError):
        degrade.safe_guard(
            lambda: (_ for _ in ()).throw(RuntimeError("must surface")),
            fallback="swallowed", name="destructive_action")


def test_all_quality_components_failopen():
    # None of these may raise on malformed input.
    qc.aggregate([None, None])
    gov.select_pipeline(None, None)
    review(None, None, None)
    degrade.guard(lambda: (_ for _ in ()).throw(ValueError()), fallback=1, name="x")
