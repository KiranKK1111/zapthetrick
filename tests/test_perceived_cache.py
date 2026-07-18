"""Predictive answer cache (perceived-speed R3, task 4.3).

Pins: exact hit + scope isolation, LRU bound, revalidate-before-serve, and
memory-vs-Redis backend parity (via a fake async Redis).
"""
from __future__ import annotations

import asyncio

from app.perceived.cache import PerceivedCache, _MemoryBackend, cache_key


class _FakeRedis:
    """Minimal async Redis stand-in (no eviction/TTL semantics needed here)."""

    def __init__(self):
        self._d = {}

    async def get(self, k):
        return self._d.get(k)

    async def set(self, k, v, ex=None):
        self._d[k] = v

    async def delete(self, k):
        self._d.pop(k, None)


class _RedisLike:
    """Adapts _FakeRedis to the cache's backend interface (pc: prefix)."""

    def __init__(self):
        self._r = _FakeRedis()

    async def get(self, key):
        return await self._r.get(f"pc:{key}")

    async def put(self, key, value):
        await self._r.set(f"pc:{key}", value)

    async def delete(self, key):
        await self._r.delete(f"pc:{key}")


def test_key_normalizes_and_scopes():
    assert cache_key("What  is HASHMAP") == cache_key("what is hashmap")
    assert cache_key("q", "user:a") != cache_key("q", "user:b")  # scope isolates


def test_exact_hit_and_miss():
    c = PerceivedCache(backend=_MemoryBackend(8))
    asyncio.run(c.put("what is a hashmap?", "A hashmap is…", scope="u1"))
    assert asyncio.run(c.get("What is a hashmap?", scope="u1")) == "A hashmap is…"
    assert asyncio.run(c.get("what is a hashmap?", scope="u2")) is None  # other scope
    assert asyncio.run(c.get("unseen", scope="u1")) is None


def test_lru_bound_evicts_oldest():
    c = PerceivedCache(backend=_MemoryBackend(2))

    async def run():
        await c.put("a", "1")
        await c.put("b", "2")
        await c.get("a")            # touch a → b is now least-recent
        await c.put("c", "3")       # evicts b
        return (await c.get("a"), await c.get("b"), await c.get("c"))

    a, b, cc = asyncio.run(run())
    assert a == "1" and cc == "3" and b is None


def test_revalidate_before_serve():
    c = PerceivedCache(backend=_MemoryBackend(8))
    asyncio.run(c.put("q", "stale answer", scope="u1"))
    # validator rejects → not served + discarded
    got = asyncio.run(c.serve_if_valid("q", "u1", validate=lambda v: "fresh" in v))
    assert got is None
    assert asyncio.run(c.get("q", "u1")) is None       # discarded
    # validator accepts → served
    asyncio.run(c.put("q", "fresh answer", scope="u1"))
    got2 = asyncio.run(c.serve_if_valid("q", "u1", validate=lambda v: "fresh" in v))
    assert got2 == "fresh answer"


def test_memory_redis_parity():
    mem = PerceivedCache(backend=_MemoryBackend(8))
    rds = PerceivedCache(backend=_RedisLike())

    async def seq(c):
        await c.put("alpha", "A", scope="u1")
        miss = await c.get("beta", scope="u1")
        hit = await c.get("alpha", scope="u1")
        wrong_scope = await c.get("alpha", scope="u2")
        return (miss, hit, wrong_scope)

    assert asyncio.run(seq(mem)) == asyncio.run(seq(rds))
