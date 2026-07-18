"""Sliding-window rate limiter + escalating cooldowns.

Ported from freellmapi's `services/ratelimit.ts`. The reference uses an
in-process SQLite DB, so all checks are synchronous and fast. We keep that
model: an **in-memory** sliding window + cooldown map is authoritative
(single uvicorn process = same semantics as the reference), and cooldowns
are additionally persisted to Postgres best-effort so a daily-quota
quarantine survives a restart.

All check functions are synchronous so the router can call them in a tight
loop without awaiting a DB round-trip per key.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

log = logging.getLogger(__name__)

_MINUTE_MS = 60_000
_HOUR_MS = 60 * _MINUTE_MS
_DAY_MS = 24 * _HOUR_MS

# Escalating cooldown: 1st 429 in 24h → 2min, then 10min, 1h, 1d+.
_COOLDOWN_DURATIONS_MS = [2 * _MINUTE_MS, 10 * _MINUTE_MS, _HOUR_MS, _DAY_MS]


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── In-memory state ──────────────────────────────────────────────────────
# request timestamps per "platform:model:key" (used for both rpm & rpd —
# pruned to the relevant window at read time).
_req_ts: dict[str, list[int]] = defaultdict(list)
# token events: (ts, tokens) per "platform:model:key".
_tok_ev: dict[str, list[tuple[int, int]]] = defaultdict(list)
# cooldown expiry ms per "platform:model:key".
_cooldowns: dict[str, int] = {}
# cooldown-set timestamps (for escalation) per "platform:model:key".
_cooldown_hits: dict[str, list[int]] = defaultdict(list)


def _k(platform: str, model_id: str, key_id: int) -> str:
    return f"{platform}:{model_id}:{key_id}"


def _prune(ts: list[int], window_ms: int, now: int) -> list[int]:
    cutoff = now - window_ms
    return [t for t in ts if t > cutoff]


# ── Request / token windows ──────────────────────────────────────────────
def can_make_request(platform: str, model_id: str, key_id: int, limits: dict) -> bool:
    now = _now_ms()
    key = _k(platform, model_id, key_id)
    ts = _req_ts[key] = _prune(_req_ts[key], _DAY_MS, now)
    rpm = limits.get("rpm")
    if rpm is not None and len([t for t in ts if t > now - _MINUTE_MS]) >= rpm:
        return False
    rpd = limits.get("rpd")
    if rpd is not None and len(ts) >= rpd:
        return False
    return True


def can_use_tokens(platform: str, model_id: str, key_id: int, estimated: int, limits: dict) -> bool:
    now = _now_ms()
    key = _k(platform, model_id, key_id)
    ev = _tok_ev[key] = [(t, n) for (t, n) in _tok_ev[key] if t > now - _DAY_MS]
    tpm = limits.get("tpm")
    if tpm is not None:
        used = sum(n for (t, n) in ev if t > now - _MINUTE_MS)
        if used + estimated > tpm:
            return False
    tpd = limits.get("tpd")
    if tpd is not None:
        used = sum(n for (t, n) in ev)
        if used + estimated > tpd:
            return False
    return True


def record_request(platform: str, model_id: str, key_id: int) -> None:
    _req_ts[_k(platform, model_id, key_id)].append(_now_ms())
    _fire(_persist_usage(platform, model_id, key_id, "request", 0))


def record_tokens(platform: str, model_id: str, key_id: int, tokens: int) -> None:
    _tok_ev[_k(platform, model_id, key_id)].append((_now_ms(), tokens))
    _fire(_persist_usage(platform, model_id, key_id, "tokens", tokens))


# ── Cooldowns ────────────────────────────────────────────────────────────
def is_on_cooldown(platform: str, model_id: str, key_id: int) -> bool:
    key = _k(platform, model_id, key_id)
    expiry = _cooldowns.get(key)
    if expiry is None:
        return False
    if _now_ms() > expiry:
        _cooldowns.pop(key, None)
        _fire(_clear_persisted_cooldown(platform, model_id, key_id))
        return False
    return True


def next_cooldown_duration(platform: str, model_id: str, key_id: int) -> int:
    """Escalating duration based on how many cooldowns this key hit in 24h."""
    key = _k(platform, model_id, key_id)
    now = _now_ms()
    hits = [t for t in _cooldown_hits[key] if t > now - _DAY_MS]
    hits.append(now)
    _cooldown_hits[key] = hits
    idx = min(len(hits) - 1, len(_COOLDOWN_DURATIONS_MS) - 1)
    return _COOLDOWN_DURATIONS_MS[idx]


def set_cooldown(platform: str, model_id: str, key_id: int, duration_ms: int | None = None) -> None:
    if duration_ms is None:
        duration_ms = next_cooldown_duration(platform, model_id, key_id)
    expiry = _now_ms() + duration_ms
    _cooldowns[_k(platform, model_id, key_id)] = expiry
    _fire(_persist_cooldown(platform, model_id, key_id, expiry))


def headroom(platform: str, model_id: str, key_id: int, limits: dict) -> float:
    """Fraction of remaining capacity for this (model, key), in [0.0, 1.0].

    1.0 = fully free (or no advertised limit); 0.0 = at the limit. Taken as the
    MIN across the rpm/rpd/tpm/tpd windows, so a model that's nearly out of any
    one budget reads as low-headroom. The router uses this to spread load: a
    near-limit model sinks below a fresher one even when it's higher in the
    manual fallback order.
    """
    now = _now_ms()
    key = _k(platform, model_id, key_id)
    ts = _prune(_req_ts.get(key, []), _DAY_MS, now)
    ev = [(t, n) for (t, n) in _tok_ev.get(key, []) if t > now - _DAY_MS]
    fracs: list[float] = []
    rpm = limits.get("rpm")
    if rpm:
        used = len([t for t in ts if t > now - _MINUTE_MS])
        fracs.append(max(0.0, (rpm - used) / rpm))
    rpd = limits.get("rpd")
    if rpd:
        fracs.append(max(0.0, (rpd - len(ts)) / rpd))
    tpm = limits.get("tpm")
    if tpm:
        used = sum(n for (t, n) in ev if t > now - _MINUTE_MS)
        fracs.append(max(0.0, (tpm - used) / tpm))
    tpd = limits.get("tpd")
    if tpd:
        used = sum(n for (t, n) in ev)
        fracs.append(max(0.0, (tpd - used) / tpd))
    return min(fracs) if fracs else 1.0


def get_status(platform: str, model_id: str, key_id: int, limits: dict) -> dict:
    now = _now_ms()
    key = _k(platform, model_id, key_id)
    ts = _prune(_req_ts.get(key, []), _DAY_MS, now)
    ev = [(t, n) for (t, n) in _tok_ev.get(key, []) if t > now - _DAY_MS]
    return {
        "rpm": {"used": len([t for t in ts if t > now - _MINUTE_MS]), "limit": limits.get("rpm")},
        "rpd": {"used": len(ts), "limit": limits.get("rpd")},
        "tpm": {"used": sum(n for (t, n) in ev if t > now - _MINUTE_MS), "limit": limits.get("tpm")},
        "on_cooldown": is_on_cooldown(platform, model_id, key_id),
    }


# ── Best-effort Postgres persistence ─────────────────────────────────────
def _fire(coro) -> None:
    """Schedule a fire-and-forget persistence coroutine if a loop is running."""
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()  # no loop (e.g. unit test) — drop it; in-memory is enough


async def _persist_usage(platform: str, model_id: str, key_id: int, kind: str, tokens: int) -> None:
    from storage.db import get_session_factory
    from storage.models import LLMRateLimitUsage

    factory = get_session_factory()
    if factory is None:
        return
    try:
        from sqlalchemy import delete

        now = _now_ms()
        async with factory() as session:
            session.add(
                LLMRateLimitUsage(
                    platform=platform, model_id=model_id, key_id=key_id,
                    kind=kind, tokens=tokens, created_at_ms=now,
                )
            )
            await session.execute(
                delete(LLMRateLimitUsage).where(LLMRateLimitUsage.created_at_ms <= now - _DAY_MS)
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        log.debug("rate-limit usage persist failed: %s", exc)


async def _persist_cooldown(platform: str, model_id: str, key_id: int, expiry_ms: int) -> None:
    from sqlalchemy.dialects.postgresql import insert

    from storage.db import get_session_factory
    from storage.models import LLMRateLimitCooldown

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as session:
            stmt = insert(LLMRateLimitCooldown).values(
                platform=platform, model_id=model_id, key_id=key_id, expires_at_ms=expiry_ms
            ).on_conflict_do_update(
                index_elements=["platform", "model_id", "key_id"],
                set_={"expires_at_ms": expiry_ms},
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("cooldown persist failed: %s", exc)


async def _clear_persisted_cooldown(platform: str, model_id: str, key_id: int) -> None:
    from sqlalchemy import delete

    from storage.db import get_session_factory
    from storage.models import LLMRateLimitCooldown

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as session:
            await session.execute(
                delete(LLMRateLimitCooldown).where(
                    LLMRateLimitCooldown.platform == platform,
                    LLMRateLimitCooldown.model_id == model_id,
                    LLMRateLimitCooldown.key_id == key_id,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("cooldown clear failed: %s", exc)


async def load_cooldowns_from_db() -> None:
    """Rehydrate in-memory cooldowns from Postgres on startup, and DELETE any
    that already expired while the process was down (they'd otherwise linger
    forever, since the lazy cleanup only fires when their exact key is next
    observed)."""
    from sqlalchemy import delete, select

    from storage.db import get_session_factory
    from storage.models import LLMRateLimitCooldown

    factory = get_session_factory()
    if factory is None:
        return
    try:
        now = _now_ms()
        async with factory() as session:
            rows = (await session.execute(select(LLMRateLimitCooldown))).scalars().all()
            for r in rows:
                if r.expires_at_ms > now:
                    _cooldowns[_k(r.platform, r.model_id, r.key_id)] = r.expires_at_ms
            # Purge already-expired rows so the table doesn't grow unbounded.
            await session.execute(
                delete(LLMRateLimitCooldown)
                .where(LLMRateLimitCooldown.expires_at_ms <= now)
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("cooldown load failed: %s", exc)


async def load_usage_from_db() -> None:
    """Rehydrate the in-memory sliding windows (request + token events) from
    the persisted usage ledger on startup. Without this, a restart wiped the
    windows and silently re-granted every provider's per-day budget (rpd/tpd),
    so the router over-drove them straight into 429s."""
    from sqlalchemy import delete, select

    from storage.db import get_session_factory
    from storage.models import LLMRateLimitUsage

    factory = get_session_factory()
    if factory is None:
        return
    try:
        now = _now_ms()
        cutoff = now - _DAY_MS
        async with factory() as session:
            # Drop stale rows first, then load what's still within the 24h window.
            await session.execute(
                delete(LLMRateLimitUsage)
                .where(LLMRateLimitUsage.created_at_ms <= cutoff)
            )
            await session.commit()
            rows = (
                await session.execute(
                    select(LLMRateLimitUsage)
                    .where(LLMRateLimitUsage.created_at_ms > cutoff)
                    .order_by(LLMRateLimitUsage.created_at_ms.asc())
                )
            ).scalars().all()
        for r in rows:
            key = _k(r.platform, r.model_id, r.key_id)
            if r.kind == "request":
                _req_ts[key].append(r.created_at_ms)
            elif r.kind == "tokens":
                _tok_ev[key].append((r.created_at_ms, r.tokens))
    except Exception as exc:  # noqa: BLE001
        log.debug("usage load failed: %s", exc)
