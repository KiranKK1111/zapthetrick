"""Providers API — manage the multi-provider LLM catalog, keys, and fallback.

This is the surface the Flutter "Providers" screen drives. It exposes the
ported freellmapi engine:

  GET    /api/providers                       -> full catalog (providers + models + key counts)
  GET    /api/providers/keys                  -> all stored keys (masked)
  POST   /api/providers/{platform}/keys       -> add a key (encrypted)
  PATCH  /api/providers/keys/{id}             -> enable/disable a key
  DELETE /api/providers/keys/{id}             -> remove a key
  GET    /api/providers/fallback              -> the priority chain
  PUT    /api/providers/fallback              -> reorder priority / enable-disable models
  POST   /api/providers/{platform}/refresh-models -> live /models discovery
  GET    /api/providers/status                -> live penalties + rate-limit snapshot
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, update

from app.llm import keys as keys_repo
from app.llm import router as llm_router
from app.llm.catalog import (
    all_providers,
    detect_moe,
    detect_param_count,
    get_provider_spec,
)
from app.llm.keys import KeysUnavailable

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/providers")


async def _models_by_platform() -> dict[str, list]:
    from storage.db import get_session_factory
    from storage.models import LLMModel

    factory = get_session_factory()
    if factory is None:
        return {}
    async with factory() as session:
        rows = (
            await session.execute(select(LLMModel).order_by(LLMModel.intelligence_rank.asc()))
        ).scalars().all()
    grouped: dict[str, list] = {}
    for m in rows:
        grouped.setdefault(m.platform, []).append(
            {
                "id": m.id,
                "model_id": m.model_id,
                "display_name": m.display_name,
                "intelligence_rank": m.intelligence_rank,
                "speed_rank": m.speed_rank,
                "size_label": m.size_label,
                "rpm_limit": m.rpm_limit,
                "rpd_limit": m.rpd_limit,
                "tpm_limit": m.tpm_limit,
                "tpd_limit": m.tpd_limit,
                "monthly_token_budget": m.monthly_token_budget,
                "context_window": m.context_window,
                "param_count": detect_param_count(m.model_id),
                "is_moe": detect_moe(m.model_id),
                "enabled": m.enabled,
                "supports_vision": getattr(m, "supports_vision", False),
            }
        )
    return grouped


@router.get("")
async def list_catalog() -> dict:
    """Full catalog: every provider with its models + key counts."""
    try:
        counts = await keys_repo.key_counts()
    except KeysUnavailable:
        counts = {}
    models = await _models_by_platform()
    providers = []
    for spec in all_providers():
        c = counts.get(spec.platform, {"total": 0, "enabled": 0, "healthy": 0})
        providers.append(
            {
                "platform": spec.platform,
                "name": spec.name,
                "base_url": spec.base_url,
                "allow_anonymous": spec.allow_anonymous,
                "keys": c,
                "models": models.get(spec.platform, []),
            }
        )
    return {"providers": providers}


@router.get("/keys")
async def list_keys(platform: str | None = None) -> dict:
    try:
        views = await keys_repo.list_keys(platform)
    except KeysUnavailable as exc:
        raise HTTPException(503, detail=str(exc))
    return {
        "keys": [
            {"id": v.id, "platform": v.platform, "label": v.label,
             "masked": v.masked, "status": v.status, "enabled": v.enabled}
            for v in views
        ]
    }


@router.post("/{platform}/keys")
async def add_key(platform: str, body: dict) -> dict:
    if get_provider_spec(platform) is None:
        raise HTTPException(404, detail=f"Unknown provider '{platform}'.")
    key = (body or {}).get("key", "")
    label = (body or {}).get("label", "")
    try:
        view = await keys_repo.add_key(platform, key, label)
    except KeysUnavailable as exc:
        raise HTTPException(503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — surface the real reason, not a bare 500
        from app.llm.crypto import EncryptionKeyError

        if isinstance(exc, EncryptionKeyError):
            raise HTTPException(
                503,
                detail="Key encryption isn't ready yet. Restart the backend, "
                "or set ZAPTHETRICK_ENCRYPTION_KEY.",
            )
        log.exception("add_key failed for %s", platform)
        raise HTTPException(500, detail=f"Could not store key: {exc}")

    # First materialize this provider's curated free-tier models (enabled,
    # with rate-limit metadata) so routing works immediately, then auto-discover
    # its full /models list (disabled) so the catalog fills out. Both
    # best-effort — a discovery hiccup must never fail the key add.
    discovered = {"discovered": 0, "added": 0}
    try:
        from app.llm.catalog import seed_provider
        from app.llm.discovery import discover_models

        await seed_provider(platform)
        discovered = await discover_models(platform, api_key=key.strip())
    except Exception as exc:  # noqa: BLE001
        log.info("seed/discovery after key add failed for %s: %s", platform, exc)

    return {
        "id": view.id, "platform": view.platform, "label": view.label,
        "masked": view.masked, "status": view.status, "enabled": view.enabled,
        "discovered": discovered,
    }


@router.patch("/keys/{key_id}")
async def patch_key(key_id: int, body: dict) -> dict:
    enabled = (body or {}).get("enabled")
    if enabled is None:
        raise HTTPException(400, detail="Body must include 'enabled': bool.")
    try:
        await keys_repo.set_enabled(key_id, bool(enabled))
    except KeysUnavailable as exc:
        raise HTTPException(503, detail=str(exc))
    return {"ok": True}


@router.post("/keys/{key_id}/validate")
async def validate_key(key_id: int) -> dict:
    """Re-validate a single key now (UI "Validate" button). Updates its status
    and re-enables it if it checks out healthy."""
    from app.llm.health import check_one_key

    status = await check_one_key(key_id)
    if status is None:
        raise HTTPException(
            404, detail="Key not found, or its provider is unavailable.")
    return {"ok": True, "id": key_id, "status": status}


@router.delete("/keys/{key_id}")
async def delete_key(key_id: int) -> dict:
    try:
        await keys_repo.delete_key(key_id)
    except KeysUnavailable as exc:
        raise HTTPException(503, detail=str(exc))
    # Drop the catalog for any provider left without a key, so its model list
    # returns to empty (matching the "no key → no models" rule).
    try:
        from app.llm.catalog import prune_keyless_providers

        await prune_keyless_providers()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True}


@router.get("/fallback")
async def get_fallback(sort: str = "manual") -> dict:
    """The priority chain, enriched with LIVE availability + capability signals
    (available-now, rate-limit headroom, free/paid, intelligence) so the UI can
    auto-prioritize. `sort=auto` returns the chain re-ordered so the best
    pickable models float to the top — WITHOUT persisting a new order (the
    user's manually-dragged `priority` is preserved)."""
    from collections import defaultdict

    from app.llm import ratelimit
    from storage.db import get_session_factory
    from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel

    factory = get_session_factory()
    if factory is None:
        raise HTTPException(503, detail="Database not ready.")
    async with factory() as session:
        rows = (
            await session.execute(
                select(LLMFallbackConfig, LLMModel)
                .join(LLMModel, LLMFallbackConfig.model_db_id == LLMModel.id)
                .order_by(LLMFallbackConfig.priority.asc())
            )
        ).all()
        keys = (
            await session.execute(
                select(LLMApiKey.platform, LLMApiKey.id).where(
                    LLMApiKey.enabled.is_(True),
                    LLMApiKey.status.in_(("healthy", "unknown")),
                )
            )
        ).all()
    penalties = {p["model_db_id"]: p["penalty"]
                 for p in llm_router.get_all_penalties()}
    key_ids: dict[str, list[int]] = defaultdict(list)
    for platform, kid in keys:
        key_ids[platform].append(kid)
    anon = {s.platform for s in all_providers()
            if getattr(s, "allow_anonymous", False)}

    def _availability(m) -> tuple[bool, float]:
        """(available-now, best rate-limit headroom 0..1) — mirrors the router's
        live eligibility (usable key, off cooldown, within request+token windows)."""
        ids = key_ids.get(m.platform) or ([0] if m.platform in anon else [])
        if not ids:
            return False, 0.0
        limits = {"rpm": m.rpm_limit, "rpd": m.rpd_limit,
                  "tpm": m.tpm_limit, "tpd": m.tpd_limit}
        ok = False
        best = 0.0
        for kid in ids:
            if (not ratelimit.is_on_cooldown(m.platform, m.model_id, kid)
                    and ratelimit.can_make_request(m.platform, m.model_id, kid, limits)
                    and ratelimit.can_use_tokens(m.platform, m.model_id, kid, 1000, limits)):
                ok = True
                try:
                    best = max(best, float(
                        ratelimit.headroom(m.platform, m.model_id, kid, limits)))
                except Exception:  # noqa: BLE001
                    best = max(best, 1.0)
        return ok, (best if ok else 0.0)

    chain = []
    for fc, m in rows:
        available, headroom = _availability(m)
        try:
            free = bool(llm_router._is_free(m, get_provider_spec(m.platform)))
        except Exception:  # noqa: BLE001
            free = False
        pen = penalties.get(fc.model_db_id, 0)
        intel = int(m.intelligence_rank or 100)
        # Composite score, LOWER = better. Unavailable models sink to the
        # bottom; among available ones: low penalty, high headroom, stronger
        # intelligence, and free tier float up.
        score = ((0.0 if available else 1000.0)
                 + pen * 4.0
                 + (1.0 - headroom) * 20.0
                 + intel * 1.0
                 + (0.0 if free else 40.0))
        chain.append({
            "model_db_id": fc.model_db_id,
            "priority": fc.priority,
            "enabled": fc.enabled,
            "platform": m.platform,
            "model_id": m.model_id,
            "display_name": m.display_name,
            "penalty": pen,
            "intelligence_rank": intel,
            "speed_rank": int(m.speed_rank or 100),
            "free": free,
            "available": available,
            "headroom": round(headroom, 3),
            "score": round(score, 2),
        })
    if (sort or "").lower() == "auto":
        chain.sort(key=lambda e: (e["score"], e["priority"]))
    return {"fallback": chain, "sort": (sort or "manual").lower()}


@router.put("/fallback")
async def set_fallback(body: dict) -> dict:
    """Reorder priorities and/or toggle models.

    Body: {"order": [model_db_id, ...]}  — index becomes priority (1-based)
          {"enabled": {model_db_id: bool, ...}}  — per-model fallback toggle
    """
    from storage.db import get_session_factory
    from storage.models import LLMFallbackConfig, LLMModel

    factory = get_session_factory()
    if factory is None:
        raise HTTPException(503, detail="Database not ready.")
    order = (body or {}).get("order") or []
    enabled = (body or {}).get("enabled") or {}
    async with factory() as session:
        for i, model_db_id in enumerate(order):
            await session.execute(
                update(LLMFallbackConfig)
                .where(LLMFallbackConfig.model_db_id == int(model_db_id))
                .values(priority=i + 1)
            )
        for model_db_id, on in enabled.items():
            mid = int(model_db_id)
            await session.execute(
                update(LLMFallbackConfig)
                .where(LLMFallbackConfig.model_db_id == mid)
                .values(enabled=bool(on))
            )
            # Keep the model's own enabled flag in sync — the router requires
            # BOTH the fallback row and the model to be enabled, so toggling
            # here is what makes a discovered/disabled model actually routable.
            await session.execute(
                update(LLMModel).where(LLMModel.id == mid).values(enabled=bool(on))
            )
        await session.commit()
    return {"ok": True}


@router.post("/{platform}/refresh-models")
async def refresh_models(platform: str) -> dict:
    """Augment the catalog with models discovered live from the provider's
    /models endpoint. New ids are added (disabled by default so they don't
    silently enter routing); known ids are left untouched."""
    from app.llm.discovery import discover_models

    if get_provider_spec(platform) is None:
        raise HTTPException(404, detail=f"Unknown provider '{platform}'.")
    result = await discover_models(platform)
    if result.get("error") and result.get("discovered", 0) == 0 and result.get("added", 0) == 0:
        raise HTTPException(400, detail=result["error"])
    return result


@router.get("/status")
async def status() -> dict:
    """Live routing health: per-model penalties for the dashboard."""
    return {"penalties": llm_router.get_all_penalties()}


async def _rotation_count(route_all: bool) -> int:
    """How many models the router can actually pick right now: models on a keyed
    (or anonymous) provider — all of them when `route_all`, else just the enabled
    fallback chain — MINUS any whose every key path is on cooldown (rate-limited
    / out-of-credits / dead). Mirrors the router's live eligibility."""
    from sqlalchemy import select

    from app.llm import ratelimit
    from storage.db import get_session_factory
    from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel

    factory = get_session_factory()
    if factory is None:
        return 0
    async with factory() as session:
        keys = (
            await session.execute(
                select(LLMApiKey.platform, LLMApiKey.id).where(
                    LLMApiKey.enabled.is_(True),
                    # Match the router: a key flagged invalid/error is NOT usable,
                    # so it shouldn't inflate the live-available count.
                    LLMApiKey.status.in_(("healthy", "unknown")),
                )
            )
        ).all()
        _cols = (LLMModel.platform, LLMModel.model_id, LLMModel.rpm_limit,
                 LLMModel.rpd_limit, LLMModel.tpm_limit, LLMModel.tpd_limit)
        if route_all:
            rows = (await session.execute(select(*_cols))).all()
        else:
            rows = (
                await session.execute(
                    select(*_cols)
                    .join(LLMFallbackConfig,
                          LLMFallbackConfig.model_db_id == LLMModel.id)
                    .where(LLMFallbackConfig.enabled.is_(True),
                           LLMModel.enabled.is_(True))
                )
            ).all()

    from collections import defaultdict
    key_ids: dict[str, list[int]] = defaultdict(list)
    for platform, kid in keys:
        key_ids[platform].append(kid)
    anon = {s.platform for s in all_providers() if getattr(s, "allow_anonymous", False)}

    live = 0
    for platform, model_id, rpm, rpd, tpm, tpd in rows:
        ids = key_ids.get(platform) or ([0] if platform in anon else [])
        if not ids:
            continue  # no usable key for this provider
        limits = {"rpm": rpm, "rpd": rpd, "tpm": tpm, "tpd": tpd}
        # Mirror the router's live eligibility: a key must be off cooldown AND
        # within its request + token windows — not just off cooldown.
        routable = any(
            (not ratelimit.is_on_cooldown(platform, model_id, kid))
            and ratelimit.can_make_request(platform, model_id, kid, limits)
            and ratelimit.can_use_tokens(platform, model_id, kid, 1000, limits)
            for kid in ids
        )
        if routable:
            live += 1
    return live


@router.get("/routing")
async def get_routing() -> dict:
    """The orchestrator's catalogue-wide routing toggle + how many models are in
    rotation right now."""
    from app.core.config_loader import get_config

    on = bool(get_config().routing.route_all_models)
    return {"route_all_models": on, "in_rotation": await _rotation_count(on)}


@router.post("/routing")
async def set_routing(body: dict) -> dict:
    """Toggle routing across EVERY configured model (incl. discovered ones).
    Live — `update_config` refreshes the cached config, no restart needed."""
    from app.core.config_loader import update_config

    on = bool(body.get("route_all_models"))
    update_config({"routing": {"route_all_models": on}})
    return {"route_all_models": on, "in_rotation": await _rotation_count(on)}
