"""Live model discovery — pull a provider's full /models list into the catalog.

Used in two places:
  * automatically right after a key is added for a provider, and
  * manually via the Providers screen's "Refresh models" button.

Discovered models are inserted **disabled** (both the `llm_models.enabled`
flag and their `llm_fallback_config` row), so they're browsable and
toggle-able without silently entering routing — the curated free-tier seed
stays the only thing enabled by default. Handles bearer providers and the
account-scoped Cloudflare endpoint.
"""
from __future__ import annotations

import logging

import httpx
from sqlalchemy import func, select

from app.llm import crypto
from app.llm.catalog import AUTH_CLOUDFLARE, get_provider_spec
from storage.db import get_session_factory
from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel

log = logging.getLogger(__name__)


async def _resolve_key(platform: str) -> str | None:
    """Decrypt the first enabled key for `platform`, or None."""
    factory = get_session_factory()
    if factory is None:
        return None
    async with factory() as session:
        row = (
            await session.execute(
                select(LLMApiKey)
                .where(LLMApiKey.platform == platform, LLMApiKey.enabled.is_(True))
                .limit(1)
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    try:
        await crypto.ensure_initialized()
        return crypto.decrypt(row.encrypted_key, row.iv, row.auth_tag)
    except Exception:  # noqa: BLE001
        return None


def _context_from_meta(meta: dict | None) -> int | None:
    """Pull a context-window length from a raw provider model object.

    OpenRouter puts it at top level (`context_length`) and sometimes under
    `top_provider.context_length`; other OpenAI-compatible providers vary.
    Returns None when absent or not a positive int."""
    if not isinstance(meta, dict):
        return None
    cw = meta.get("context_length") or meta.get("context_window")
    if not cw:
        tp = meta.get("top_provider")
        if isinstance(tp, dict):
            cw = tp.get("context_length")
    try:
        cw = int(cw)
        return cw if cw > 0 else None
    except (TypeError, ValueError):
        return None


def _build_request(spec, api_key: str | None) -> tuple[str, dict] | None:
    """Return (url, headers) for the provider's /models endpoint, or None."""
    if spec.auth == AUTH_CLOUDFLARE:
        if not api_key or ":" not in api_key:
            return None
        account_id, token = api_key.split(":", 1)
        url = spec.base_url.format(account_id=account_id).rstrip("/") + "/models"
        return url, {"Authorization": f"Bearer {token}"}
    url = spec.base_url.rstrip("/") + "/models"
    headers = dict(spec.extra_headers)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return url, headers


async def discover_models(platform: str, api_key: str | None = None) -> dict:
    """Fetch `platform`'s /models and add any new ones (disabled).

    Returns {discovered, added, error?}. Never raises for ordinary failures
    (no key, provider down) — callers treat it as best-effort.
    """
    spec = get_provider_spec(platform)
    if spec is None:
        return {"discovered": 0, "added": 0, "error": f"unknown provider '{platform}'"}
    factory = get_session_factory()
    if factory is None:
        return {"discovered": 0, "added": 0, "error": "database not ready"}

    if api_key is None:
        api_key = await _resolve_key(platform)
    if not api_key and not spec.allow_anonymous:
        return {"discovered": 0, "added": 0, "error": "no API key for this provider"}

    req = _build_request(spec, api_key)
    if req is None:
        return {"discovered": 0, "added": 0, "error": "could not build request (bad key?)"}
    url, headers = req

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return {"discovered": 0, "added": 0, "error": f"could not reach provider: {exc}"}
    if resp.status_code != 200:
        return {"discovered": 0, "added": 0, "error": f"provider returned {resp.status_code}"}

    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001 — some endpoints serve HTML/empty
        return {"discovered": 0, "added": 0, "error": f"non-JSON /models response: {exc}"}

    # Providers vary: OpenAI shape `{data:[{id}]}`, a bare list of objects, or
    # even a bare list of model-id strings. Handle all three.
    if isinstance(body, list):
        data = body
    elif isinstance(body, dict):
        data = body.get("data") or body.get("models") or []
    else:
        data = []
    # Keep id → metadata so we can detect vision capability per model.
    meta_by_id: dict[str, dict] = {}
    ids: list[str] = []
    for m in data:
        if isinstance(m, str):
            ids.append(m)
        elif isinstance(m, dict):
            mid = m.get("id") or m.get("name")
            if mid:
                ids.append(mid)
                meta_by_id[mid] = m

    from app.llm.catalog import detect_vision, rank_from_id

    added = 0
    async with factory() as session:
        existing = {
            m.model_id: m
            for m in (
                await session.execute(select(LLMModel).where(LLMModel.platform == platform))
            ).scalars().all()
        }
        max_priority = (
            await session.execute(select(func.max(LLMFallbackConfig.priority)))
        ).scalar() or 0
        for mid in ids:
            vision = detect_vision(mid, meta_by_id.get(mid))
            if mid in existing:
                # Keep an already-known model's vision flag fresh (metadata may
                # now say it's multimodal even if it was added id-only before).
                if vision and not existing[mid].supports_vision:
                    existing[mid].supports_vision = True
                # Backfill context window if discovery now carries it.
                if existing[mid].context_window is None:
                    ctx = _context_from_meta(meta_by_id.get(mid))
                    if ctx:
                        existing[mid].context_window = ctx
                continue
            intel, speed = rank_from_id(mid, meta_by_id.get(mid))
            model = LLMModel(
                platform=platform, model_id=mid, display_name=mid,
                intelligence_rank=intel, speed_rank=speed, enabled=False,
                supports_vision=vision,
                context_window=_context_from_meta(meta_by_id.get(mid)),
            )
            session.add(model)
            await session.flush()  # assign id
            max_priority += 1
            session.add(
                LLMFallbackConfig(model_db_id=model.id, priority=max_priority, enabled=False)
            )
            existing[mid] = model
            added += 1
        await session.commit()

    if added:
        log.info("discovery: %s +%d models (of %d listed)", platform, added, len(ids))
    return {"discovered": len(ids), "added": added}
