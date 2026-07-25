"""Liveness + readiness probes (vNext §6.2).

Splits health into two honest signals:

* ``/healthz`` (liveness) — the process is up and answering. Cheap, never
  flaps. This is what the pod watchdog / RunPod HTTP proxy should poll.
* ``/readyz`` (readiness) — every subsystem the app needs to actually WORK is
  warm: Postgres reachable, the embedder loaded, the STT chain warm, and
  (optionally, behind flags) the vision VLM and an LLM route.

Why the gating defaults are what they are: a fresh pod serves the UI and accepts
provider-key uploads with **zero** cloud keys configured, so "ready to work" ==
the *local* subsystems being warm (PG + embedder + STT). That is exactly the
moment the RunPod proxy should go green — and gating on it fixes the
"stuck Initializing / watchdog kills the app mid-warmup" behavior, because the
watchdog polls ``/healthz`` (always up) while ``/readyz`` reports the real state.
Router/vision gating flips on later behind ``cfg.health.gate_router`` /
``cfg.health.gate_vision`` once the local T4 floor (§2.1) lands.

Every check is isolated and fail-open: a probe that errors reports ``ok=None``
(unknown) with a detail string rather than throwing. A **gated** unknown counts
as not-ready (honest/conservative); a non-gated check is reported but never
blocks readiness.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def liveness() -> dict:
    """Process-up signal. Intentionally does nothing expensive."""
    return {"status": "ok"}


def _flag(name: str, default: bool) -> bool:
    """Read ``cfg.health.<name>`` defensively (section may not exist yet)."""
    try:
        from app.core.config_loader import cfg
        health = getattr(cfg, "health", None)
        if health is None:
            return default
        v = getattr(health, name, default)
        return default if v is None else bool(v)
    except Exception:  # noqa: BLE001 — never let a config read break a probe
        return default


def _check_postgres() -> tuple[bool | None, str]:
    """Postgres reachable + migrations landed."""
    try:
        from storage import bootstrap as _bs
        if not bool(getattr(_bs, "POSTGRES_READY", False)):
            return False, "Postgres not ready"
        state = getattr(_bs, "MIGRATION_STATE", "ready")
        if state and state != "ready":
            return False, f"migrations {state}"
        return True, "reachable"
    except Exception as exc:  # noqa: BLE001
        return None, f"probe error: {exc}"


def _warmup_snapshot() -> dict | None:
    try:
        from app import models_warmup as _mw
        return _mw.snapshot()
    except Exception as exc:  # noqa: BLE001
        log.debug("readiness: warmup snapshot failed: %s", exc)
        return None


def _row(snapshot: dict | None, pred) -> dict | None:
    if not snapshot:
        return None
    for m in snapshot.get("models", []) or []:
        try:
            if pred(m):
                return m
        except Exception:  # noqa: BLE001
            continue
    return None


def _check_embedder(snapshot: dict | None) -> tuple[bool | None, str]:
    if snapshot is None:
        return None, "warmup status unavailable"
    row = _row(snapshot, lambda m: m.get("key") == "embedder")
    if row is None:
        # No embedder row registered — warmup either disabled or already cached
        # before it ran. Treat overall all_ready as the fallback signal.
        return (True if snapshot.get("all_ready") else None), "no embedder row"
    stage = row.get("stage")
    if stage == "ready":
        return True, "loaded"
    if stage == "error":
        return False, str(row.get("detail") or "load error")
    return False, f"stage={stage}"


def _check_stt(snapshot: dict | None) -> tuple[bool | None, str]:
    if snapshot is None:
        return None, "warmup status unavailable"
    # The PRIMARY stt row is the first-registered `stt:*`; fallbacks may be
    # skipped without harming readiness.
    row = _row(snapshot, lambda m: str(m.get("key", "")).startswith("stt:"))
    if row is None:
        return (True if snapshot.get("all_ready") else None), "no stt row"
    stage = row.get("stage")
    if stage in ("ready", "skipped"):
        return True, f"primary {stage}"
    if stage == "error":
        return False, str(row.get("detail") or "load error")
    return False, f"stage={stage}"


def _check_vision(snapshot: dict | None) -> tuple[bool | None, str]:
    # Soft by default (non-gating in Stage 1). If vision is disabled, it's a
    # non-issue; if enabled but no warmup row is registered, we can't confirm.
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(getattr(cfg, "vision", None), "enabled", True)):
            return True, "vision disabled"
    except Exception:  # noqa: BLE001
        pass
    row = _row(snapshot, lambda m: m.get("key") == "vision")
    if row is None:
        return None, "no vision warmup row"
    return (row.get("stage") == "ready"), f"stage={row.get('stage')}"


async def _check_router() -> tuple[bool | None, str]:
    """Best-effort: is at least one LLM route usable right now? Non-gating in
    Stage 1 (flips gating on once the local T4 floor guarantees a route). Never
    raises; unknown on any error or when Postgres isn't ready."""
    try:
        from storage import bootstrap as _bs
        if not bool(getattr(_bs, "POSTGRES_READY", False)):
            return None, "db not ready"
        from sqlalchemy import func, select
        from storage.db import get_session_factory
        from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel
        factory = get_session_factory()
        if factory is None:
            return None, "no session factory"
        async with factory() as session:
            enabled = (await session.execute(
                select(func.count()).select_from(LLMFallbackConfig)
                .where(LLMFallbackConfig.enabled.is_(True))
            )).scalar() or 0
            healthy_keys = (await session.execute(
                select(func.count()).select_from(LLMApiKey)
                .where(LLMApiKey.enabled.is_(True),
                       LLMApiKey.status.in_(("healthy", "unknown")))
            )).scalar() or 0
        if enabled and healthy_keys:
            return True, f"{enabled} routes, {healthy_keys} keys"
        # Enabled models but no keys → the user still needs to add provider keys
        # via the UI. Honest 'unknown' (not a hard failure) so a keyless fresh
        # pod is still live and can accept the upload.
        return None, f"{enabled} routes, {healthy_keys} healthy keys"
    except Exception as exc:  # noqa: BLE001
        return None, f"probe error: {exc}"


async def readiness_report() -> dict:
    """Aggregate readiness across subsystems. See module docstring for gating."""
    snap = _warmup_snapshot()

    pg_ok, pg_detail = _check_postgres()
    emb_ok, emb_detail = _check_embedder(snap)
    stt_ok, stt_detail = _check_stt(snap)
    vis_ok, vis_detail = _check_vision(snap)
    try:
        router_ok, router_detail = await _check_router()
    except Exception as exc:  # noqa: BLE001
        router_ok, router_detail = None, f"probe error: {exc}"

    gate_vision = _flag("gate_vision", False)
    gate_router = _flag("gate_router", False)

    checks = {
        "postgres": {"ok": pg_ok, "gate": True, "detail": pg_detail},
        "embedder": {"ok": emb_ok, "gate": True, "detail": emb_detail},
        "stt": {"ok": stt_ok, "gate": True, "detail": stt_detail},
        "vision": {"ok": vis_ok, "gate": gate_vision, "detail": vis_detail},
        "router": {"ok": router_ok, "gate": gate_router, "detail": router_detail},
    }
    # Ready == every GATED check is explicitly True. A gated unknown (None) or
    # False blocks readiness.
    ready = all(c["ok"] is True for c in checks.values() if c["gate"])
    return {"ready": ready, "checks": checks}


__all__ = ["liveness", "readiness_report"]
