"""Architecture Health Dashboard (P1 #6) — composed snapshot, fail-open."""
from __future__ import annotations

from app.obs import health_dashboard


def test_snapshot_has_all_sections():
    snap = health_dashboard.snapshot()
    for key in ("cache", "providers", "jobs", "failures"):
        assert key in snap


def test_providers_reflect_quota_manager():
    snap = health_dashboard.snapshot()
    # quota_manager seeds known free providers.
    assert isinstance(snap["providers"], list)


def test_jobs_section_counts_running():
    from app.obs.jobs import jobs
    jid = jobs().start("dashboard-test", kind="task")
    snap = health_dashboard.snapshot()
    assert snap["jobs"].get("running", 0) >= 1
    jobs().finish(jid, ok=True)


def test_snapshot_has_new_phase1_fields():
    """P1 #6: router cost, retrieval relevance, verifier failure rate."""
    snap = health_dashboard.snapshot()
    for key in ("router_cost", "retrieval_relevance", "verifier"):
        assert key in snap, f"missing dashboard field {key}"


def test_retrieval_relevance_reflects_recorded_scores():
    from app.obs import metrics
    metrics.reset_health_counters()
    metrics.record_retrieval_relevance(0.9)
    metrics.record_retrieval_relevance(0.7)
    snap = health_dashboard.snapshot()["retrieval_relevance"]
    assert snap["samples"] == 2
    assert abs(snap["avg_relevance"] - 0.8) < 1e-6


def test_router_cost_reflects_recorded_calls():
    from app.obs import metrics
    metrics.reset_health_counters()
    metrics.record_router_cost(tokens=100, cost_usd=0.002)
    metrics.record_router_cost(tokens=50)
    snap = health_dashboard.snapshot()["router_cost"]
    assert snap["calls"] == 2
    assert snap["est_tokens"] == 150


def test_verifier_failure_rate():
    from app.obs import metrics
    metrics.reset_health_counters()
    metrics.record_verify(True)
    metrics.record_verify(True)
    metrics.record_verify(False)
    snap = health_dashboard.snapshot()["verifier"]
    assert snap["verified"] == 3
    assert snap["failures"] == 1
    assert abs(snap["failure_rate"] - round(1 / 3, 4)) < 1e-4


def test_verifier_folds_in_artifact_validation():
    """Verifier rate is genuinely populated from the already-wired artifact
    validation ledger, even with no direct verify feed."""
    from app.obs import metrics
    from app.obs import decision_metrics as dm
    metrics.reset_health_counters()
    dm.reset_for_tests()
    dm.record_artifact_validation({"validated": True})
    dm.record_artifact_validation({"validated": False})   # failed
    snap = health_dashboard.snapshot()["verifier"]
    assert snap.get("artifact_total") == 2
    assert snap.get("artifact_failures") == 1


def test_routes_import():
    import app.api.routes_jobs as m
    assert m.health_router is not None
