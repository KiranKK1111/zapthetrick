"""Wiring test — the Recovery Planner (Phase 4 #8) is USED live in the LLM
engine retry loop, consuming the classified failure to log a budgeted recovery
decision. Static source assertion + a live behavioral check.
"""
from __future__ import annotations

import pathlib

from app.llm.providers import ProviderError
from app.obs import classify_exception, plan_recovery

_ENGINE = pathlib.Path(__file__).resolve().parents[1] / "app" / "llm" / "engine.py"


def test_engine_wires_recovery_planner():
    src = _ENGINE.read_text(encoding="utf-8")
    assert "from app.obs import recovery as _recovery" in src
    assert "_recovery.plan_recovery(" in src
    assert "attempt=_attempt + 1" in src  # budgeted, not blind


def test_recovery_flows_from_engine_style_failure():
    # The exact chain the engine runs: classify a provider error, plan recovery.
    fc = classify_exception(ProviderError("HTTP 429 rate limit"))
    plan = plan_recovery(fc, attempt=1, max_attempts=6)
    assert plan.failure_id == "provider_rate_limit"
    assert plan.should_retry and plan.backoff_ms > 0
    # Budget exhaustion stops the retry (no blind loop).
    assert not plan_recovery(fc, attempt=6, max_attempts=6).should_retry
