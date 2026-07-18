"""Shared pooled HTTP client (perceived-speed R2, task 1.3).

Pins: one client reused across calls, rebuilt on dispose (config change),
disposed (closed) on shutdown, and a per-request fallback when the pool build
fails (R2.3).
"""
from __future__ import annotations

import asyncio

import httpx

from app.core import http_pool


def _reset(monkeypatch):
    monkeypatch.setattr(http_pool, "_shared", None, raising=False)


def test_get_http_client_is_reused(monkeypatch):
    _reset(monkeypatch)
    c1 = http_pool.get_http_client()
    c2 = http_pool.get_http_client()
    try:
        assert c1 is c2                       # one shared, pooled client
        assert isinstance(c1, httpx.AsyncClient)
    finally:
        asyncio.run(http_pool.dispose_http_client())


def test_rebuilt_after_dispose(monkeypatch):
    _reset(monkeypatch)
    c1 = http_pool.get_http_client()
    asyncio.run(http_pool.dispose_http_client())   # e.g. on LLM config change
    assert http_pool._shared is None
    assert c1.is_closed
    c2 = http_pool.get_http_client()
    try:
        assert c2 is not c1                   # a fresh client after rebuild
    finally:
        asyncio.run(http_pool.dispose_http_client())


def test_dispose_is_idempotent(monkeypatch):
    _reset(monkeypatch)
    # No client built yet — dispose must be a safe no-op.
    asyncio.run(http_pool.dispose_http_client())
    asyncio.run(http_pool.dispose_http_client())
    assert http_pool._shared is None


def test_per_request_fallback_on_build_error(monkeypatch):
    _reset(monkeypatch)

    def _boom():
        raise RuntimeError("pool build failed (e.g. bad limits)")

    monkeypatch.setattr(http_pool, "_build", _boom)
    c = http_pool.get_http_client()
    try:
        # Still functional (a fresh per-request client) …
        assert isinstance(c, httpx.AsyncClient)
        # … and the shared slot stays empty so a later call can retry pooling.
        assert http_pool._shared is None
    finally:
        asyncio.run(c.aclose())
