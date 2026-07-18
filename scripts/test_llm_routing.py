"""Standalone self-test for the multi-provider routing engine.

Runs without pytest, Postgres, or real API keys — it exercises the pure
in-memory logic (crypto, rate-limit windows + cooldowns, penalty/decay) and
the engine's fallback loop using a fake router + fake adapter.

    python scripts/test_llm_routing.py

Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import sys

# Encryption key must exist before importing crypto-using paths.
os.environ.setdefault("ZAPTHETRICK_ENCRYPTION_KEY", secrets.token_hex(32))

# Make `app` importable when run from the backend root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm import crypto, engine, ratelimit, router  # noqa: E402
from app.llm.providers.base import ProviderError  # noqa: E402


def test_crypto() -> None:
    asyncio.run(crypto.init_encryption_key())
    secret = "sk-or-v1-" + secrets.token_hex(16)
    enc, iv, tag = crypto.encrypt(secret)
    assert crypto.decrypt(enc, iv, tag) == secret, "crypto round-trip mismatch"
    assert crypto.mask_key(secret).endswith(secret[-4:]), "mask should keep last 4"
    print("  [ok] crypto round-trip + mask")


def test_ratelimit() -> None:
    p, m, k = "groq", "llama-3.3-70b-versatile", 1
    limits = {"rpm": 2, "rpd": None, "tpm": None, "tpd": None}
    assert ratelimit.can_make_request(p, m, k, limits)
    ratelimit.record_request(p, m, k)
    ratelimit.record_request(p, m, k)
    assert not ratelimit.can_make_request(p, m, k, limits), "rpm=2 should be exhausted"

    # Token window
    tlim = {"tpm": 100, "tpd": None}
    assert ratelimit.can_use_tokens(p, m, 2, 50, tlim)
    ratelimit.record_tokens(p, m, 2, 80)
    assert not ratelimit.can_use_tokens(p, m, 2, 50, tlim), "80+50 > 100 tpm"

    # Cooldown set + manual expiry
    ratelimit.set_cooldown(p, m, 3, duration_ms=10_000)
    assert ratelimit.is_on_cooldown(p, m, 3)
    ratelimit._cooldowns[ratelimit._k(p, m, 3)] = ratelimit._now_ms() - 1  # force-expire
    assert not ratelimit.is_on_cooldown(p, m, 3), "expired cooldown should clear"
    print("  [ok] rate-limit windows + cooldown")


def test_penalty_decay() -> None:
    mid = 42
    assert router.get_penalty(mid) == 0
    router.record_rate_limit_hit(mid)
    router.record_rate_limit_hit(mid)
    assert router.get_penalty(mid) == 6, router.get_penalty(mid)
    # Force decay: backdate last_hit by > one interval.
    router._penalties[mid]["last_hit"] -= router._DECAY_INTERVAL_S + 1
    assert router.get_penalty(mid) == 5, "one point should decay"
    router.record_success(mid)
    assert router.get_penalty(mid) == 4
    print("  [ok] penalty + time decay")


def test_engine_fallback() -> None:
    """Fake a chain [A(429), B(ok)] and assert the engine falls through to B
    and penalizes A."""
    routes = [
        router.RouteResult("groq", "model-A", 101, "Model A", "keyA", 11),
        router.RouteResult("cerebras", "model-B", 102, "Model B", "keyB", 22),
    ]
    calls = {"route": 0}

    async def fake_route(est, skip=None, preferred=None):
        skip = skip or set()
        for r in routes:
            if f"{r.platform}:{r.model_id}:{r.key_id}" in skip:
                continue
            calls["route"] += 1
            return r
        raise router.NoRouteAvailable("exhausted")

    class FakeAdapter:
        def __init__(self, model_id):
            self.model_id = model_id

        async def complete(self, api_key, messages, model_id, options):
            if model_id == "model-A":
                raise ProviderError("rate limited", status=429)
            return "hello from B"

    def fake_get_adapter(platform):
        return FakeAdapter("model-A" if platform == "groq" else "model-B")

    orig_route, orig_adapter = router.route_request, engine.get_adapter
    engine.router.route_request = fake_route
    engine.get_adapter = fake_get_adapter
    try:
        before = router.get_penalty(101)
        text, route = asyncio.run(
            engine.route_and_complete([{"role": "user", "content": "hi"}], {})
        )
        assert text == "hello from B", text
        assert route.model_db_id == 102, "should land on model B"
        assert router.get_penalty(101) > before, "model A should be penalized"
        assert calls["route"] == 2, "should have routed twice (A then B)"
        print("  [ok] engine fallback A(429) -> B + penalty recorded")
    finally:
        engine.router.route_request = orig_route
        engine.get_adapter = orig_adapter


def test_dead_model_classification() -> None:
    """The two real-world OpenRouter failures must be retryable + dead."""
    from app.llm.providers.base import classify_error

    r, d = classify_error(404, "OpenRouter API error 404: No endpoints found for moonshotai/kimi-k2:free")
    assert r and d, "404 no-endpoints must be retryable+dead"
    r, d = classify_error(400, "OpenRouter API error 400: deepseek/deepseek-v3.1:free is not a valid model ID")
    assert r and d, "400 invalid-model must be retryable+dead"
    r, d = classify_error(401, "OpenRouter API error 401: invalid api key")
    assert not r and not d, "401 auth must NOT be retryable"
    r, d = classify_error(429, "Groq API error 429: rate limit")
    assert r and not d, "429 retryable but not dead"
    print("  [ok] dead/auth/rate-limit error classification")


def test_engine_dead_fallback() -> None:
    """A dead model (invalid id) must be skipped and the chain continued."""
    routes = [
        router.RouteResult("openrouter", "deepseek/deepseek-v3.1:free", 201, "DeepSeek V3.1 (free)", "k", 1),
        router.RouteResult("openrouter", "qwen/qwen3-coder:free", 202, "Qwen3 Coder (free)", "k", 1),
    ]

    async def fake_route(est, skip=None, preferred=None):
        skip = skip or set()
        for r in routes:
            if f"{r.platform}:{r.model_id}:{r.key_id}" in skip:
                continue
            return r
        raise router.NoRouteAvailable("exhausted")

    class FakeAdapter:
        async def complete(self, api_key, messages, model_id, options):
            if model_id == "deepseek/deepseek-v3.1:free":
                raise ProviderError(
                    "OpenRouter API error 400: deepseek/deepseek-v3.1:free is not a valid model ID",
                    status=400,
                )
            return "answer from qwen"

    orig_route, orig_adapter = router.route_request, engine.get_adapter
    engine.router.route_request = fake_route
    engine.get_adapter = lambda platform: FakeAdapter()
    try:
        text, route = asyncio.run(
            engine.route_and_complete([{"role": "user", "content": "hi"}], {})
        )
        assert text == "answer from qwen", text
        assert route.model_id == "qwen/qwen3-coder:free", "should skip dead model"
        print("  [ok] engine skips dead model (400 invalid id) -> next")
    finally:
        engine.router.route_request = orig_route
        engine.get_adapter = orig_adapter


def main() -> None:
    print("LLM routing self-test")
    test_crypto()
    test_ratelimit()
    test_penalty_decay()
    test_engine_fallback()
    test_dead_model_classification()
    test_engine_dead_fallback()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
