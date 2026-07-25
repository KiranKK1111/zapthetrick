"""Provider pre-connect (vNext §3.4) — keep warm HTTP/2 pools to the top-N
candidate providers the router favors right now, so a turn (and its hedge)
fires on an already-open connection instead of paying DNS + TCP + TLS
(~100-300 ms) before the first request byte.

The connection pool is the process-wide `app.core.http_pool` client; warming =
issuing one cheap `HEAD` to each candidate host so the pooled keepalive socket
is established ahead of the real request. Even a 4xx/405 response (or a failed
request) is useful — the TCP+TLS handshake still completed, leaving a live
pooled connection.

Fire-and-forget + debounced (`schedule()`), and **fail-open**: off by default
(`resilience.pre_connect`), and any error is swallowed so a pool hiccup can
never touch a turn. It makes hedging (§2.2) cheaper by construction — the hedge
fires on an open connection instead of paying its own handshake.

Candidate set = the current user's configured+enabled providers first, then
anonymous-tier providers, deduped by connection host and capped at top-N — i.e.
exactly the hosts the router chooses among for this account.
"""
from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Per-user-scope debounce timestamps (hosts differ per account).
_last_warm: dict[str, float] = {}


def _cfg() -> tuple[bool, int, float]:
    """(on, top_n, min_interval_s) — defensive, never raises out."""
    try:
        from app.core.config_loader import cfg
        r = cfg.resilience
        return (bool(getattr(r, "pre_connect", False)),
                int(getattr(r, "pre_connect_top_n", 3)),
                float(getattr(r, "pre_connect_min_interval_s", 30.0)))
    except Exception:  # noqa: BLE001
        return (False, 3, 30.0)


def enabled() -> bool:
    return _cfg()[0]


def _host_base(base_url: str) -> str | None:
    """`scheme://netloc` of a provider base_url — the actual connection target.
    A templated host (e.g. cloudflare's `{account_id}` URL) has no fixed host to
    warm, so it's skipped."""
    try:
        if not base_url or "{" in base_url:
            return None
        u = urlparse(base_url)
        if not u.scheme or not u.netloc:
            return None
        return f"{u.scheme}://{u.netloc}"
    except Exception:  # noqa: BLE001
        return None


async def _candidate_bases(top_n: int) -> list[str]:
    """Distinct provider connection hosts the router could pick right now: the
    user's configured+enabled providers first, then anonymous-tier providers;
    deduped by host, capped at `top_n`."""
    from app.llm import catalog
    platforms: list[str] = []
    try:
        from app.llm import keys
        for k in await keys.list_keys():
            if getattr(k, "enabled", True) and k.platform not in platforms:
                platforms.append(k.platform)
    except Exception:  # noqa: BLE001 — no keys / DB down → anonymous set only
        pass
    try:
        for spec in catalog.all_providers():
            if getattr(spec, "allow_anonymous", False) \
                    and spec.platform not in platforms:
                platforms.append(spec.platform)
    except Exception:  # noqa: BLE001
        pass
    bases: list[str] = []
    seen: set[str] = set()
    cap = max(1, top_n)
    for plat in platforms:
        spec = catalog.get_provider_spec(plat)
        if spec is None:
            continue
        host = _host_base(spec.base_url)
        if host and host not in seen:
            seen.add(host)
            bases.append(host)
        if len(bases) >= cap:
            break
    return bases


async def warm(*, top_n: int | None = None, force: bool = False) -> int:
    """Establish a pooled keepalive connection to each top-N candidate host so
    the next real request skips the handshake. Bounded, best-effort, fail-open;
    returns how many hosts were reached (a failed reach may still have warmed
    the socket, so this is a floor, not the true warmth count). `force=True`
    warms even when `resilience.pre_connect` is off (the §3.10 input-warmup lane
    drives it on its own flag)."""
    on, cfg_n, _ = _cfg()
    if not on and not force:
        return 0
    n = top_n if top_n is not None else cfg_n
    try:
        bases = await _candidate_bases(n)
    except Exception:  # noqa: BLE001
        return 0
    if not bases:
        return 0
    try:
        from app.core.http_pool import get_http_client
        client = get_http_client()
    except Exception:  # noqa: BLE001
        return 0

    warmed = 0

    async def _one(base: str) -> None:
        nonlocal warmed
        try:
            await client.head(base, timeout=3.0)
            warmed += 1
        except Exception:  # noqa: BLE001 — the handshake likely completed anyway
            pass

    await asyncio.gather(*[_one(b) for b in bases], return_exceptions=True)
    if warmed:
        log.debug("preconnect: warmed %d/%d provider host(s)", warmed, len(bases))
    return warmed


def schedule() -> None:
    """Fire-and-forget debounced warm. No-op when pre-connect is off, when there
    is no running event loop, or within the per-user min-interval. Safe to call
    at the start of every turn — the debounce prevents spam."""
    on, _, min_interval = _cfg()
    if not on:
        return
    try:
        from storage.context import get_request_user_id
        who = str(get_request_user_id() or "anon")
    except Exception:  # noqa: BLE001
        who = "anon"
    now = time.monotonic()
    if now - _last_warm.get(who, 0.0) < min_interval:
        return
    _last_warm[who] = now
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(warm())


def reset_for_tests() -> None:
    _last_warm.clear()


__all__ = ["warm", "schedule", "enabled", "reset_for_tests"]
