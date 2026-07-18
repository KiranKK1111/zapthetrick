"""Tests for the two built-out Phase 4 gaps:
  #8  Recovery Planner        (app/obs/recovery.py)
  #18 Failure Prediction      (app/obs/failure_prediction.py)
Both consume the Phase 1 failure taxonomy. Deterministic + offline.
"""
from __future__ import annotations

from app.obs import failure_taxonomy as ft
from app.obs import plan_recovery, predict
from app.obs.failure_taxonomy import Recovery


# ── #8 recovery planner ─────────────────────────────────────────────────────
def test_rate_limit_gets_escalating_backoff():
    p1 = plan_recovery("provider_rate_limit", attempt=1)
    p2 = plan_recovery("provider_rate_limit", attempt=2)
    assert p1.action == Recovery.COOLDOWN_WAIT.value
    assert p1.should_retry and p2.should_retry
    assert p2.backoff_ms > p1.backoff_ms  # escalating


def test_retry_budget_is_bounded_no_blind_retry():
    # At the last attempt, a retryable failure must STOP retrying (rule #10).
    p = plan_recovery("provider_transport", attempt=3, max_attempts=3)
    assert not p.should_retry
    assert "exhausted" in p.rationale


def test_terminal_failure_never_retries():
    p = plan_recovery("provider_exhausted", attempt=1)  # recovery=ESCALATE
    assert not p.should_retry
    assert p.action == Recovery.ESCALATE.value
    assert p.backoff_ms == 0


def test_plan_from_exception_and_id_and_class():
    from_exc = plan_recovery(TimeoutError("timed out"), attempt=1)
    assert from_exc.failure_id == "generation_timeout"
    from_id = plan_recovery("verification_failed", attempt=1)
    assert from_id.action == Recovery.REPAIR.value and from_id.should_retry
    from_class = plan_recovery(ft.get("network_error"), attempt=1)
    assert from_class.failure_id == "network_error"  # DEGRADE → terminal
    assert not from_class.should_retry


def test_recovery_never_raises():
    p = plan_recovery(object(), attempt=1)  # bogus input
    assert p.failure_id == "internal_error"


# ── #18 failure prediction ──────────────────────────────────────────────────
def test_predicts_network_failure_offline():
    r = predict(needs_network=True, network_available=False)
    assert r.risky
    top = r.top()
    assert top.failure_id == "network_error" and top.likelihood >= 0.9


def test_predicts_timeout_on_huge_input():
    r = predict(input_chars=5_000_000, max_input_chars=200_000)
    assert r.risky
    assert r.top().failure_id == "generation_timeout"


def test_predicts_missing_sdk():
    r = predict(needs_sdk="rustc", available_sdks={"python", "node"})
    assert any(p.failure_id == "verification_failed" for p in r.predictions)


def test_clean_task_is_not_risky():
    r = predict(needs_network=True, network_available=True, input_chars=500,
                needs_sdk="python", available_sdks={"python"})
    assert not r.risky
    assert r.predictions == []


def test_predictions_are_taxonomy_aligned():
    r = predict(needs_network=True, network_available=False, input_chars=10_000_000)
    for p in r.predictions:
        assert ft.get(p.failure_id) is not None  # every prediction maps to a real class
