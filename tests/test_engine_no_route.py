"""Engine backs off + retries route selection on a transient NoRouteAvailable
(concurrent-burst rate-limit exhaustion) instead of erroring on the first raise."""
from __future__ import annotations

import asyncio

import app.llm.engine as E
from app.llm import router as R


async def _drain(agen):
    out = []
    async for x in agen:
        out.append(x)
    return out


def test_transient_no_route_is_retried_with_backoff(monkeypatch):
    calls = {"n": 0}

    async def fake_route(*a, **k):
        calls["n"] += 1
        # TRANSIENT: candidates exist but are all momentarily rate-limited.
        raise R.NoRouteAvailable("all models exhausted", transient=True)

    async def no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E.asyncio, "sleep", no_sleep)     # keep the test instant
    monkeypatch.setattr(E, "_no_route_retry", lambda: (2, 0.0))  # 2 retries
    monkeypatch.setattr(E, "_max_retries", lambda: 6)

    async def run():
        await _drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s1"))

    # A transient exhaustion retries: 1 initial + 2 backoff retries = 3 calls,
    # then propagates.
    try:
        asyncio.run(run())
        assert False, "expected NoRouteAvailable to propagate after retries"
    except R.NoRouteAvailable:
        pass
    assert calls["n"] == 3


def test_persistent_no_route_fails_fast(monkeypatch):
    calls = {"n": 0}

    async def fake_route(*a, **k):
        calls["n"] += 1
        # PERSISTENT (default): no key / no provider — retrying is pointless.
        raise R.NoRouteAvailable("no provider configured")

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E, "_no_route_retry", lambda: (2, 0.0))  # retries allowed
    monkeypatch.setattr(E, "_max_retries", lambda: 6)

    async def run():
        await _drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s3"))

    try:
        asyncio.run(run())
        assert False, "expected NoRouteAvailable"
    except R.NoRouteAvailable:
        pass
    assert calls["n"] == 1   # fast-fail — no wasted retries/backoff


def test_no_route_disabled_errors_immediately(monkeypatch):
    calls = {"n": 0}

    async def fake_route(*a, **k):
        calls["n"] += 1
        raise R.NoRouteAvailable("all models exhausted")

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E, "_no_route_retry", lambda: (0, 0.0))  # disabled
    monkeypatch.setattr(E, "_max_retries", lambda: 6)

    async def run():
        await _drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s2"))

    try:
        asyncio.run(run())
        assert False, "expected NoRouteAvailable"
    except R.NoRouteAvailable:
        pass
    assert calls["n"] == 1   # no retries → single attempt
