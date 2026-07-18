"""Redis / DragonflyDB adapter — same wire protocol, same code.

DragonflyDB advertises itself as Redis-compatible; the same
`redis.asyncio` client talks to both. We use that fact to keep this
adapter unified instead of writing two near-identical files.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator


class RedisCache:
    """Async Redis cache + pub/sub.

    Lazy connection: the first method call opens the pool. We don't
    fail import-time if `redis` isn't installed — it surfaces only
    on first use.
    """

    def __init__(self, *, url: str, default_ttl_seconds: int = 3600) -> None:
        self.url = url
        self.default_ttl = default_ttl_seconds
        self._client = None
        self._lock = asyncio.Lock()

    async def _get(self):
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    import redis.asyncio as aioredis
                except ImportError as exc:
                    raise RuntimeError(
                        "redis is not installed. Run: pip install redis"
                    ) from exc
                self._client = aioredis.from_url(
                    self.url,
                    decode_responses=True,
                )
        return self._client

    # ---- key/value ---------------------------------------------------
    async def get(self, key: str) -> str | None:
        client = await self._get()
        v = await client.get(key)
        return v if v is not None else None

    async def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        client = await self._get()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        await client.set(key, value, ex=ttl if ttl > 0 else None)

    async def delete(self, key: str) -> None:
        client = await self._get()
        await client.delete(key)

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        client = await self._get()
        n = await client.incr(key)
        if ttl_seconds is not None:
            await client.expire(key, ttl_seconds)
        return int(n)

    # ---- pub / sub ---------------------------------------------------
    async def publish(self, channel: str, message: str) -> None:
        client = await self._get()
        await client.publish(channel, message)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        client = await self._get()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for msg in pubsub.listen():
                # Drop the initial "subscribe" confirmation.
                if msg.get("type") == "message":
                    data = msg.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", "replace")
                    yield str(data)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
