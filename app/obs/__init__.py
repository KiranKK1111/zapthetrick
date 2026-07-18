"""Observability — per-run metrics for the chat agent loop (Phase 9, #12/#29).

Reuses the existing `agent_runs` ledger table (no new migration): one row per
chat agent-run records latency, tokens (estimate), tool calls, repair rounds,
verification + confidence, and success — for a metrics view and cost/latency
analytics.
"""
from .failure_taxonomy import (
    FailureClass,
    Recovery,
    Severity,
    classify_exception,
    observe,
)
from .failure_kb import best_recovery, record_occurrence, record_outcome
from .failure_prediction import PreflightReport, RiskPrediction, predict
from .metrics import RunMetrics, aggregate_runs, est_tokens
from .recovery import RecoveryPlan, plan_recovery
from .replay import Recording, ReplayReport, ReplayStore, record

__all__ = [
    "RunMetrics", "aggregate_runs", "est_tokens",
    "FailureClass", "Recovery", "Severity", "classify_exception", "observe",
    "Recording", "ReplayStore", "ReplayReport", "record",
    "RecoveryPlan", "plan_recovery",
    "PreflightReport", "RiskPrediction", "predict",
    "record_occurrence", "record_outcome", "best_recovery",
]
