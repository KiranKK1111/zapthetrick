"""T5 exhaustion fallback — serve a recent cached answer when all routes are
down (vNext §2.1, the ladder's last rung).
"""
from __future__ import annotations

import asyncio

from app.core import llm_client as LC
from app.llm import cache as C
from app.llm.router import NoRouteAvailable


def test_relaxed_key_is_prompt_only_and_deterministic():
    msgs = [{"role": "user", "content": "explain X"}]
    k = C.relaxed_key(msgs)
    assert k is not None
    assert k == C.relaxed_key(msgs)          # deterministic
    # It ignores answer-shaping options by construction (it takes only messages),
    # so the same prompt maps to the same relaxed key regardless of difficulty.
    assert C.relaxed_key(msgs, model="a") != C.relaxed_key(msgs, model="b")
    # Multimodal / empty → not cacheable.
    assert C.relaxed_key([{"role": "user", "content": [{"type": "image"}]}]) is None


def test_t5_serves_cache_on_exhaustion(monkeypatch):
    C.clear()
    msgs = [{"role": "user", "content": "what is 2+2"}]
    # A prior answer was cached under the relaxed (prompt-only) key...
    C.put(C.relaxed_key(msgs), "four (from a recent answer)")

    client = LC.LLMClient()

    async def _boom(*a, **k):
        raise NoRouteAvailable("all providers down", transient=False)

    monkeypatch.setattr(client, "_auto_complete", _boom)

    text, mid = asyncio.run(client.complete_routed(msgs))
    assert text == "four (from a recent answer)"
    assert mid is None


def test_t5_reraises_when_no_cache(monkeypatch):
    C.clear()
    msgs = [{"role": "user", "content": "a brand new question"}]
    client = LC.LLMClient()

    async def _boom(*a, **k):
        raise NoRouteAvailable("all providers down", transient=False)

    monkeypatch.setattr(client, "_auto_complete", _boom)

    # No cached answer for this prompt → the routing error still surfaces.
    try:
        asyncio.run(client.complete_routed(msgs))
        assert False, "expected NoRouteAvailable"
    except NoRouteAvailable:
        pass


def test_successful_call_populates_both_keys(monkeypatch):
    C.clear()
    msgs = [{"role": "user", "content": "cache me"}]
    client = LC.LLMClient()

    class _Route:
        model_db_id = 5

    async def _ok(*a, **k):
        return "the answer", _Route()

    monkeypatch.setattr(client, "_auto_complete", _ok)
    text, mid = asyncio.run(client.complete_routed(msgs))
    assert text == "the answer"
    assert mid == 5
    # Both the exact and the relaxed key are now warm for future T5 fallbacks.
    assert C.get(C.relaxed_key(msgs)) == "the answer"
