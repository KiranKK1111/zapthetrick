"""Background key health checker.

Ported from freellmapi's `services/health.ts`. Every 5 minutes, validate
each enabled key against its provider:

  * confirmed 401     → status='invalid' (out of rotation, but kept enabled so
                        the next sweep re-checks it and a fixed key self-heals).
  * 403 / transport   → inconclusive: preserve a prior 'healthy' status (a blip
                        must not drop a working key); else 'error'.
  * otherwise         → status='healthy', reset fail_count.

Keys are NEVER auto-disabled by the health check — a disabled key leaves the
sweep permanently and can never recover on its own.

Started from the app lifespan via `start_health_loop()`.
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.llm import crypto
from app.llm.providers import ProviderError, get_adapter
from storage.db import get_session_factory
from storage.models import LLMApiKey

log = logging.getLogger(__name__)

CHECK_INTERVAL_S = 5 * 60
_task: asyncio.Task | None = None


async def check_all_keys() -> int:
    """Validate every enabled key once. Returns the number checked."""
    factory = get_session_factory()
    if factory is None:
        return 0
    try:
        await crypto.ensure_initialized()
    except Exception:  # noqa: BLE001 — keys just read as undecryptable below
        pass
    async with factory() as session:
        keys = (
            await session.execute(select(LLMApiKey).where(LLMApiKey.enabled.is_(True)))
        ).scalars().all()
        checked = 0
        for key in keys:
            adapter = get_adapter(key.platform)
            if adapter is None:
                continue
            try:
                plain = crypto.decrypt(key.encrypted_key, key.iv, key.auth_tag)
            except Exception:  # noqa: BLE001
                key.status = "error"
                continue
            checked += 1
            try:
                ok = await adapter.validate_key(plain)
            except Exception:  # noqa: BLE001
                # Transport / inconclusive (incl. 403) — the check told us
                # NOTHING about the key. Do NOT overwrite a previously-healthy
                # status: a momentary network blip at sweep time must not drop a
                # working key out of rotation (the router excludes non-healthy).
                # Only stamp 'error' when we had no prior good result.
                if (key.status or "unknown") not in ("healthy", "unknown"):
                    key.status = "error"
                elif key.status is None:
                    key.status = "unknown"
                continue
            if ok:
                key.status = "healthy"
                key.fail_count = 0
            else:
                # Authoritative bad key (401). Mark invalid — the router already
                # excludes non-healthy keys, so it drops out of rotation — but do
                # NOT auto-disable: a disabled key leaves the sweep forever and
                # can never self-heal. Keeping it enabled+invalid means the next
                # sweep re-checks it and a fixed/rotated key recovers on its own.
                key.status = "invalid"
                key.fail_count = (key.fail_count or 0) + 1
            from sqlalchemy import func

            key.last_checked_at = func.now()
        await session.commit()
        return checked


async def check_one_key(key_id: int) -> str | None:
    """Validate a single key on demand (powers the UI "Validate" button).

    Returns the new status ('healthy' | 'invalid' | 'error'), or None if the
    key/provider is missing. A successful check re-enables an auto-disabled
    key and resets its fail count.
    """
    factory = get_session_factory()
    if factory is None:
        return None
    try:
        await crypto.ensure_initialized()
    except Exception:  # noqa: BLE001 — the decrypt below reports "error"
        pass
    async with factory() as session:
        key = await session.get(LLMApiKey, key_id)
        if key is None:
            return None
        adapter = get_adapter(key.platform)
        if adapter is None:
            return None
        try:
            plain = crypto.decrypt(key.encrypted_key, key.iv, key.auth_tag)
        except Exception:  # noqa: BLE001
            key.status = "error"
            await session.commit()
            return "error"
        try:
            ok = await adapter.validate_key(plain)
        except ProviderError:
            key.status = "error"
            await session.commit()
            return "error"
        except Exception:  # noqa: BLE001
            key.status = "error"
            await session.commit()
            return "error"
        from sqlalchemy import func

        if ok:
            key.status = "healthy"
            key.fail_count = 0
            key.enabled = True  # a manual re-validate revives a disabled key
        else:
            key.status = "invalid"
            key.fail_count = (key.fail_count or 0) + 1
        key.last_checked_at = func.now()
        await session.commit()
        return key.status


async def _loop() -> None:
    # Initial delay so startup + migrations settle first.
    await asyncio.sleep(15)
    while True:
        try:
            n = await check_all_keys()
            if n:
                log.info("llm health: validated %d key(s)", n)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — never let the loop die
            log.warning("llm health check failed: %s", exc)
        await asyncio.sleep(CHECK_INTERVAL_S)


def start_health_loop() -> None:
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(), name="llm-health-loop")


def stop_health_loop() -> None:
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None
