"""Never-empty route ladder (vNext §2.1).

Pins select_route()'s T0–T3 ladder with a STUBBED route_request (no DB). The
ladder is flag-gated: OFF → delegate + raise exactly as today; ON → widen
through the rungs and, when exhausted, return route=None WITHOUT raising.
"""
from __future__ import annotations

import asyncio

import pytest

from app.llm import router as R


def _route(mid=1):
    return R.RouteResult("plat", "model", mid, "Model", "key", 1)


def _install(monkeypatch, fn, *, ladder: bool):
    monkeypatch.setattr(R, "route_request", fn)
    monkeypatch.setattr(R, "_ladder_enabled", lambda: ladder)


def test_flag_off_delegates_and_raises(monkeypatch):
    async def boom(**kw):
        raise R.NoRouteAvailable("x", transient=True)
    _install(monkeypatch, boom, ladder=False)
    with pytest.raises(R.NoRouteAvailable):
        asyncio.run(R.select_route())


def test_flag_off_delegates_success_is_T1(monkeypatch):
    async def ok(**kw):
        return _route(3)
    _install(monkeypatch, ok, ladder=False)
    d = asyncio.run(R.select_route())
    assert d.route.model_db_id == 3
    assert d.rung == "T1"


def test_T0_sticky_when_preferred_is_returned(monkeypatch):
    async def ok(**kw):
        return _route(7)
    _install(monkeypatch, ok, ladder=True)
    d = asyncio.run(R.select_route(preferred_model_db_id=7))
    assert d.rung == "T0"


def test_T1_when_full_filters_succeed(monkeypatch):
    async def ok(**kw):
        return _route(1)
    _install(monkeypatch, ok, ladder=True)
    d = asyncio.run(R.select_route(needs_json=True))
    assert d.rung == "T1"
    assert d.degraded == []


def test_T2_relaxes_capability_filters(monkeypatch):
    async def stub(**kw):
        if kw.get("needs_tool") or kw.get("needs_json"):
            raise R.NoRouteAvailable("filtered", transient=False)
        return _route(2)
    _install(monkeypatch, stub, ladder=True)
    d = asyncio.run(R.select_route(needs_json=True))
    assert d.rung == "T2"
    assert d.degraded == ["tier_fallback"]


def test_T3_relaxes_min_context_floor(monkeypatch):
    # The genuine rescue: an over-large request pruned every model on min_context
    # at T1/T2; T3 drops the floor so it still routes to the best available.
    async def stub(**kw):
        if kw.get("needs_tool") or kw.get("needs_json") or kw.get("min_context"):
            raise R.NoRouteAvailable("too big / filtered", transient=False)
        return _route(9)
    _install(monkeypatch, stub, ladder=True)
    d = asyncio.run(R.select_route(needs_json=True, min_context=1_000_000))
    assert d.rung == "T3"
    assert d.degraded == ["tier_fallback", "wide_fallback"]
    assert d.route.model_db_id == 9


def test_exhausted_returns_none_without_raising(monkeypatch):
    async def always(**kw):
        raise R.NoRouteAvailable("nothing", transient=True)
    _install(monkeypatch, always, ladder=True)
    d = asyncio.run(R.select_route())
    assert d.route is None
    assert d.rung == "exhausted"
    assert d.transient is True
    assert d.degraded == ["route_exhausted"]
