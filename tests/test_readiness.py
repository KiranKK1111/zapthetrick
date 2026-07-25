"""Liveness + readiness probes (vNext §6.2).

Pins: liveness is trivial; readiness gates on the LOCAL subsystems (Postgres +
embedder + STT) and is fail-open on the non-gated router/vision checks. Async
report driven via asyncio.run so the suite needs no asyncio-mode config.
"""
from __future__ import annotations

import asyncio

from app.api import readiness as R


def _fake_snapshot(*, embedder="ready", stt="ready", all_ready=True):
    models = []
    if embedder is not None:
        models.append({"key": "embedder", "stage": embedder})
    if stt is not None:
        models.append({"key": "stt:parakeet", "stage": stt})
    return {"models": models, "all_ready": all_ready}


def test_liveness_is_trivially_ok():
    assert R.liveness() == {"status": "ok"}


def test_ready_when_local_subsystems_warm(monkeypatch):
    # A truly-ready pod has Postgres reachable AND migrations landed (the app's
    # own contract: data routes 503 until MIGRATION_STATE == "ready").
    monkeypatch.setattr("storage.bootstrap.POSTGRES_READY", True, raising=False)
    monkeypatch.setattr("storage.bootstrap.MIGRATION_STATE", "ready", raising=False)
    monkeypatch.setattr("app.models_warmup.snapshot", _fake_snapshot)
    rep = asyncio.run(R.readiness_report())
    assert rep["ready"] is True
    assert rep["checks"]["postgres"]["ok"] is True
    assert rep["checks"]["embedder"]["ok"] is True
    assert rep["checks"]["stt"]["ok"] is True
    # router/vision are non-gating in Stage 1 and must not block readiness.
    assert rep["checks"]["router"]["gate"] is False
    assert rep["checks"]["vision"]["gate"] is False


def test_not_ready_when_embedder_errored(monkeypatch):
    monkeypatch.setattr("storage.bootstrap.POSTGRES_READY", True, raising=False)
    monkeypatch.setattr(
        "app.models_warmup.snapshot",
        lambda: _fake_snapshot(embedder="error", all_ready=False),
    )
    rep = asyncio.run(R.readiness_report())
    assert rep["ready"] is False
    assert rep["checks"]["embedder"]["ok"] is False


def test_not_ready_when_postgres_down(monkeypatch):
    monkeypatch.setattr("storage.bootstrap.POSTGRES_READY", False, raising=False)
    monkeypatch.setattr("app.models_warmup.snapshot", _fake_snapshot)
    rep = asyncio.run(R.readiness_report())
    assert rep["ready"] is False
    assert rep["checks"]["postgres"]["ok"] is False


def test_stt_skipped_primary_counts_as_ready(monkeypatch):
    # A primary STT that reports 'skipped' (e.g. provider not installed) must
    # not block readiness — it's an honest terminal state.
    monkeypatch.setattr("storage.bootstrap.POSTGRES_READY", True, raising=False)
    monkeypatch.setattr(
        "app.models_warmup.snapshot",
        lambda: _fake_snapshot(stt="skipped"),
    )
    rep = asyncio.run(R.readiness_report())
    assert rep["checks"]["stt"]["ok"] is True


def test_endpoints_are_wired():
    # /healthz and /readyz must be registered on the app (no lifespan run).
    from app.main import app
    paths = {r.path for r in app.routes}
    assert "/healthz" in paths
    assert "/readyz" in paths
