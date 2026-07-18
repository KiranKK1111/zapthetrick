"""Shared pooled HTTP client (perceived-speed R2).

A single process-wide ``httpx.AsyncClient`` with keep-alive + HTTP/2 so provider
connections (DNS + TLS + handshake) are reused across requests instead of being
rebuilt on every call. This removes ~100-500ms of connection setup from the
critical path — the single biggest "actual latency" win after streaming.

Callers pass a per-request ``timeout=`` to the actual request call so the varied
timeouts the old per-request clients used (5s health, 60s chat, …) are preserved
even though the shared client has one default timeout.

Fail-open (R2.3): if building the pooled client fails — e.g. the optional ``h2``
package is missing for HTTP/2, or any other error — callers fall back to a fresh
per-request client so behavior is never worse than today. Idle keepalive
connections are released after ``cfg.perceived.connection_idle_timeout_s`` (R2.2).
"""
from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

# Process-wide shared client. Rebuilt lazily; dropped on config change/shutdown.
# An httpx.AsyncClient binds its connection pool to the event loop it first
# connects on, so we ALSO track that loop and rebuild whenever the current
# running loop differs (or the old loop closed). Without this, a client built on
# a throwaway loop — e.g. a startup warmup thread's `asyncio.run(...)` — poisons
# every later call on the main server loop with "Event loop is closed".
_shared: httpx.AsyncClient | None = None
_shared_loop: asyncio.AbstractEventLoop | None = None


def _idle_expiry() -> float:
    """Keepalive idle expiry (seconds) — released idle warmed connections."""
    try:
        from app.core.config_loader import cfg
        per = getattr(cfg, "perceived", None)
        return float(getattr(per, "connection_idle_timeout_s", 60.0) or 60.0)
    except Exception:  # noqa: BLE001 — never let config break the pool
        return 60.0


def _default_timeout() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.llm, "timeout_seconds", 60) or 60)
    except Exception:  # noqa: BLE001
        return 60.0


def _build() -> httpx.AsyncClient:
    """Build the pooled client. Prefers HTTP/2; falls back to HTTP/1.1 when the
    optional ``h2`` package isn't installed."""
    limits = httpx.Limits(
        max_keepalive_connections=50,
        max_connections=200,
        keepalive_expiry=_idle_expiry(),
    )
    timeout = _default_timeout()
    try:
        return httpx.AsyncClient(http2=True, timeout=timeout, limits=limits)
    except Exception as exc:  # noqa: BLE001 — h2 missing, etc.
        log.info("http_pool: HTTP/2 unavailable (%s) — using HTTP/1.1 pool", exc)
        return httpx.AsyncClient(timeout=timeout, limits=limits)


def get_http_client() -> httpx.AsyncClient:
    """Return the shared pooled client, building it lazily.

    On any build failure, return a fresh unpooled client so the caller still
    works (per-request fallback, R2.3); the shared slot stays empty so a later
    call can retry building the pool.
    """
    global _shared, _shared_loop
    try:
        try:
            cur = asyncio.get_running_loop()
        except RuntimeError:
            cur = None
        c = _shared
        # Rebuild when: no client yet / it was closed / it was built on a
        # DIFFERENT loop / that loop has since closed. The stale client is
        # abandoned — its loop is gone, so it can't be aclose()d anyway.
        _stale_loop = (_shared_loop is not cur) or (
            _shared_loop is not None and _shared_loop.is_closed())
        if c is None or c.is_closed or _stale_loop:
            c = _build()
            _shared = c
            _shared_loop = cur
        return c
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "http_pool: pooled client unavailable (%s) — per-request fallback", exc
        )
        try:
            return httpx.AsyncClient(timeout=_default_timeout())
        except Exception:  # noqa: BLE001
            return httpx.AsyncClient()


async def dispose_http_client() -> None:
    """Close + drop the shared client so the next ``get_http_client()`` rebuilds
    it with current config (used on LLM config change and on shutdown)."""
    global _shared, _shared_loop
    c = _shared
    _shared = None
    _shared_loop = None
    if c is not None and not c.is_closed:
        try:
            await c.aclose()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["get_http_client", "dispose_http_client"]
