"""The reliability loop is CLOSED in the engine (audit fixes).

Two loops were open:
  1. the failure KB accumulated occurrences but was never told whether a recovery
     WORKED (`record_outcome` was test-only) → `best_recovery` always None;
  2. `plan_recovery` was computed and then ignored — the retry gate keyed on
     `exc.retryable` alone, so the classified/budgeted strategy changed nothing.

These tests drive the real engine retry loop (fake router + fake adapter) and
assert both loops are now closed.
"""
from __future__ import annotations

import asyncio

import pytest

import app.llm.engine as E
from app.llm import router as R
from app.llm.providers import ProviderError
from app.obs import failure_kb as kb
from app.obs.recovery import RecoveryPlan

_TEXT = "A kube-proxy watches the API server and programs iptables rules. " \
        "It keeps service virtual IPs reachable from every node in the cluster."


class _Route:
    """Minimal stand-in for router.RouteResult."""

    def __init__(self, i: int) -> None:
        self.platform = "fake"
        self.model_id = f"m{i}"
        self.key_id = i
        self.api_key = "k"
        self.model_db_id = i
        self.display_name = f"Model {i}"


class _Adapter:
    """Adapter whose per-attempt behaviour is scripted: an Exception instance in
    `script` is raised, a string is streamed/returned."""

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls = 0

    def _next(self):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return item

    async def stream(self, api_key, messages, model_id, options):
        item = self._next()
        if isinstance(item, Exception):
            raise item
        yield item

    async def complete(self, api_key, messages, model_id, options):
        item = self._next()
        if isinstance(item, Exception):
            raise item
        return item


def _wire(monkeypatch, script: list, *, retries: int = 3):
    """Fake router + adapter; returns (adapter, routes_used, sleeps)."""
    routes: list[int] = []
    sleeps: list[float] = []
    adapter = _Adapter(script)

    async def fake_route(*_a, **_k):
        routes.append(len(routes) + 1)
        return _Route(len(routes))

    async def fake_sleep(secs=0, *_a, **_k):
        sleeps.append(secs)

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E, "get_adapter", lambda _p: adapter)
    monkeypatch.setattr(E.asyncio, "sleep", fake_sleep)   # keep the test instant
    monkeypatch.setattr(E, "_max_retries", lambda: retries)
    monkeypatch.setattr(E, "_first_token_deadline", lambda: 0.0)
    return adapter, routes, sleeps


async def _drain(agen) -> str:
    out = []
    async for x in agen:
        out.append(x)
    return "".join(out)


@pytest.fixture(autouse=True)
def _clean_kb():
    kb.reset()
    yield
    kb.reset()


# ── BUG 1: the KB now LEARNS (record_outcome fires in production) ────────────

def test_record_outcome_on_success_after_retry(monkeypatch):
    """Streaming: 429 on model 1, clean answer on model 2 → the KB is told the
    recovery WORKED (this is what makes best_recovery return anything)."""
    _wire(monkeypatch, [ProviderError("HTTP 429 too many requests"), _TEXT])

    text = asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert text == _TEXT

    acts = kb.stats()["provider_rate_limit"]["actions"]
    assert "cooldown_wait" in acts, "the action actually taken must be recorded"
    assert acts["cooldown_wait"]["attempts"] == 1
    # A success was recorded, so the smoothed rate is above the 0.5 prior.
    assert acts["cooldown_wait"]["success_rate"] > 0.5


def test_record_outcome_on_exhausted_retries(monkeypatch):
    """Non-streaming: every attempt fails → the KB is told the recovery FAILED."""
    _wire(monkeypatch, [ProviderError("connection reset by peer")], retries=2)

    with pytest.raises(R.NoRouteAvailable):
        asyncio.run(E.route_and_complete(
            [{"role": "user", "content": "hi"}], {}, session_key="s"))

    acts = kb.stats()["provider_transport"]["actions"]
    assert acts["retry_different"]["attempts"] == 2   # one per failed attempt
    assert acts["retry_different"]["success_rate"] < 0.5   # all failures


def test_kb_learns_a_best_recovery_end_to_end(monkeypatch):
    """The whole point: after real traffic, `best_recovery` stops returning None."""
    assert kb.best_recovery("provider_rate_limit") is None
    for _ in range(2):
        _wire(monkeypatch, [ProviderError("HTTP 429 rate limit"), _TEXT])
        asyncio.run(_drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert kb.best_recovery("provider_rate_limit") == "cooldown_wait"


def test_no_outcome_recorded_on_a_clean_first_attempt(monkeypatch):
    """No failure → nothing to learn (don't poison the KB with phantom wins)."""
    _wire(monkeypatch, [_TEXT])
    asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert kb.stats() == {}


# ── BUG 2: the retry gate HONOURS the plan ──────────────────────────────────

def test_retry_gate_consults_the_plan_not_just_retryable(monkeypatch):
    """A TERMINAL plan stops the loop even though `exc.retryable` is True — proof
    the gate keys on the recovery plan and not on the exception flag alone."""
    _, routes, _ = _wire(monkeypatch, [ProviderError("HTTP 429 rate limit")],
                         retries=4)

    def terminal_plan(_failure, *, attempt=1, max_attempts=3):
        return RecoveryPlan("provider_exhausted", "escalate", False, attempt,
                            max_attempts, 0, "terminal")

    monkeypatch.setattr(E._recovery, "plan_recovery", terminal_plan)

    exc = ProviderError("HTTP 429 rate limit")
    assert exc.retryable, "fixture sanity: the old gate WOULD have retried"

    with pytest.raises(ProviderError):
        asyncio.run(_drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert len(routes) == 1, "terminal plan → no fallback attempts"


def test_retryable_plan_still_falls_back(monkeypatch):
    """Control for the test above: the same error with the real (non-terminal)
    plan does fall through to the next model."""
    _, routes, _ = _wire(monkeypatch,
                         [ProviderError("HTTP 429 rate limit"), _TEXT], retries=4)
    asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert len(routes) == 2


def test_plan_backoff_is_applied(monkeypatch):
    """The plan's backoff is actually slept (429 → cooldown_wait → 1s on attempt 1).
    Before the fix nothing ever slept — the plan was logging-only."""
    _, _, sleeps = _wire(monkeypatch,
                         [ProviderError("HTTP 429 rate limit"), _TEXT])
    monkeypatch.setattr(E, "_recovery_backoff_cap_ms", lambda: 1000)
    asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert sleeps == [1.0]


def test_plan_backoff_is_capped(monkeypatch):
    """…and bounded by the cap, so a rate limit can't stall the live hot path."""
    _, _, sleeps = _wire(monkeypatch,
                         [ProviderError("HTTP 429 rate limit"), _TEXT])
    monkeypatch.setattr(E, "_recovery_backoff_cap_ms", lambda: 200)
    asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert sleeps == [0.2], "the cap bounds the planner's escalating cooldown"


def test_plan_routes_to_a_different_model(monkeypatch):
    """RETRY_DIFFERENT (transport error) → the failed MODEL is passed to the router
    as `avoid_model_db_id`, not merely the (model,key) pair skipped."""
    seen: list[int | None] = []
    adapter = _Adapter([ProviderError("connection reset by peer"), _TEXT])

    async def fake_route(*_a, **k):
        seen.append(k.get("avoid_model_db_id"))
        return _Route(len(seen))

    async def fake_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E, "get_adapter", lambda _p: adapter)
    monkeypatch.setattr(E.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(E, "_max_retries", lambda: 3)
    monkeypatch.setattr(E, "_first_token_deadline", lambda: 0.0)

    asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert seen == [None, 1], "the 2nd attempt must avoid the model that failed"


def test_mid_stream_failure_is_never_re_routed(monkeypatch):
    """`yielded` stays an absolute veto — the plan can't re-route once bytes are
    on the wire."""

    class _MidFail:
        async def stream(self, *_a, **_k):
            yield _TEXT
            raise ProviderError("HTTP 429 rate limit")

    async def fake_route(*_a, **_k):
        return _Route(1)

    monkeypatch.setattr(R, "route_request", fake_route)
    monkeypatch.setattr(E, "get_adapter", lambda _p: _MidFail())
    monkeypatch.setattr(E, "_max_retries", lambda: 3)
    monkeypatch.setattr(E, "_first_token_deadline", lambda: 0.0)

    with pytest.raises(ProviderError):
        asyncio.run(_drain(E.route_and_stream(
            [{"role": "user", "content": "hi"}], {}, session_key="s")))


# ── Fail-open: bookkeeping must never break the LLM call ─────────────────────

def test_planner_blowup_falls_back_to_retryable_gate(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("planner exploded")

    _wire(monkeypatch, [ProviderError("HTTP 429 rate limit"), _TEXT])
    monkeypatch.setattr(E._recovery, "plan_recovery", boom)

    text = asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert text == _TEXT   # the call still succeeds via `exc.retryable`


def test_kb_blowup_does_not_break_the_call(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("kb exploded")

    _wire(monkeypatch, [ProviderError("HTTP 429 rate limit"), _TEXT])
    monkeypatch.setattr(E._failure_kb, "record_outcome", boom)
    monkeypatch.setattr(E._failure_kb, "record_occurrence", boom)

    text = asyncio.run(_drain(E.route_and_stream(
        [{"role": "user", "content": "hi"}], {}, session_key="s")))
    assert text == _TEXT
