"""Proactive dead-model reaper.

The request-time router already fails over a gone/EOL model and prunes it on
first hit (`classify_error` → `permanent_dead` → `engine._disable_model`). This
reaper does that PROACTIVELY, on a slow cadence, so a decommissioned model never
even costs one failed round-trip mid-conversation.

Cheap + last-known-good-safe, two stages:
  1. Per platform with an enabled key, fetch the live `/models` id set ONCE
     (`discovery.fetch_model_ids`). A model still listed is alive → skip.
  2. Only for an ENABLED routing model the provider has quietly DROPPED from a
     SUCCESSFUL, non-empty list, confirm with a minimal 1-token completion. Prune
     (delete from the catalog) ONLY when that probe returns a confirmed dead/EOL
     error (`ProviderError.permanent_dead` — 410 Gone, "end of life", invalid id).

Never prunes on a failed/None/empty list fetch, nor on a rate-limit / transport /
paywall error — those aren't "dead". So a provider's bad `/models` day or a busy
free tier can't delete a healthy model (mirrors discovery's §2.8 invariant).
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.llm import crypto, discovery
from app.llm.providers import ProviderError, get_adapter
from storage.db import get_session_factory
from storage.models import LLMApiKey, LLMFallbackConfig, LLMModel

log = logging.getLogger(__name__)

CHECK_INTERVAL_S = 30 * 60  # slow cadence — dead models are rare, probes cost quota
_PROBE_CONCURRENCY = 4
_task: asyncio.Task | None = None


async def _decrypt_first_key(platform: str) -> str | None:
    """Decrypt the first enabled key for `platform` (for the probe call)."""
    factory = get_session_factory()
    if factory is None:
        return None
    async with factory() as session:
        row = (
            await session.execute(
                select(LLMApiKey).where(
                    LLMApiKey.platform == platform,
                    LLMApiKey.enabled.is_(True),
                    LLMApiKey.status.in_(("healthy", "unknown")),
                ).limit(1)
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    try:
        await crypto.ensure_initialized()
        return crypto.decrypt(row.encrypted_key, row.iv, row.auth_tag)
    except Exception:  # noqa: BLE001
        return None


async def _enabled_routing_models() -> list[tuple[int, str, str]]:
    """(model_db_id, platform, model_id) for every ENABLED routing model."""
    factory = get_session_factory()
    if factory is None:
        return []
    async with factory() as session:
        rows = (
            await session.execute(
                select(LLMModel.id, LLMModel.platform, LLMModel.model_id)
                .join(LLMFallbackConfig,
                      LLMFallbackConfig.model_db_id == LLMModel.id)
                .where(LLMModel.enabled.is_(True),
                       LLMFallbackConfig.enabled.is_(True))
            )
        ).all()
    return [(r[0], r[1], r[2]) for r in rows]


async def _probe_is_dead(platform: str, model_id: str, api_key: str) -> bool:
    """True ONLY when a minimal completion confirms the model is permanently
    gone. Any other outcome (success, rate-limit, transport, paywall) → False."""
    adapter = get_adapter(platform)
    if adapter is None:
        return False
    try:
        await adapter.complete(
            api_key,
            [{"role": "user", "content": "ping"}],
            model_id,
            {"max_tokens": 1, "temperature": 0},
        )
        return False  # answered → alive
    except ProviderError as exc:
        return bool(exc.permanent_dead)
    except Exception:  # noqa: BLE001 — a flaky probe is never grounds to delete
        return False


async def reap_dead_models() -> int:
    """One reaper pass. Returns the number of models pruned from the catalog."""
    models = await _enabled_routing_models()
    if not models:
        return 0

    # Group by platform so each provider's /models + key are fetched once.
    by_platform: dict[str, list[tuple[int, str]]] = {}
    for model_db_id, platform, model_id in models:
        by_platform.setdefault(platform, []).append((model_db_id, model_id))

    from app.llm.engine import _disable_model  # lazy: engine is heavy

    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    pruned = 0

    for platform, entries in by_platform.items():
        live_ids = await discovery.fetch_model_ids(platform)
        if not live_ids:
            continue  # unknown / empty list → prune nothing (last-known-good)
        suspicious = [(mid_db, mid) for (mid_db, mid) in entries
                      if mid not in live_ids]
        if not suspicious:
            continue
        api_key = await _decrypt_first_key(platform)
        if not api_key:
            continue  # can't confirm without a key → leave it alone

        async def _check(model_db_id: int, model_id: str) -> int:
            async with sem:
                dead = await _probe_is_dead(platform, model_id, api_key)
            if dead:
                await _disable_model(model_db_id)
                log.info("reaper: pruned dead model %s/%s (id=%s)",
                         platform, model_id, model_db_id)
                return 1
            return 0

        results = await asyncio.gather(
            *(_check(mid_db, mid) for (mid_db, mid) in suspicious),
            return_exceptions=True,
        )
        pruned += sum(r for r in results if isinstance(r, int))

    return pruned


async def _loop() -> None:
    await asyncio.sleep(60)  # let startup, migrations, and discovery settle
    while True:
        try:
            n = await reap_dead_models()
            if n:
                log.info("reaper: pruned %d dead model(s)", n)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — never let the loop die
            log.warning("dead-model reaper failed: %s", exc)
        await asyncio.sleep(CHECK_INTERVAL_S)


def start_reaper_loop() -> None:
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(), name="llm-model-reaper")


def stop_reaper_loop() -> None:
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None
