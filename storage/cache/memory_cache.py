"""In-process fallback cache. No external service required.

Used in tests and as a graceful degrade when Dragonfly / Redis isn't
reachable. Honors TTL via expiration timestamps; pub/sub uses
[asyncio.Queue] per channel.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator


class MemoryCache:
    def __init__(self, *, default_ttl_seconds: int = 3600) -> None:
        self.default_ttl = default_ttl_seconds
        self._store: dict[str, tuple[str, float | None]] = {}
        self._counters: dict[str, int] = {}
        self._channels: dict[str, list[asyncio.Queue[str]]] = {}

    # ---- key/value ---------------------------------------------------
    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.monotonic() + ttl if ttl > 0 else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._counters.pop(key, None)

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        n = self._counters.get(key, 0) + 1
        self._counters[key] = n
        return n

    # ---- pub / sub ---------------------------------------------------
    async def publish(self, channel: str, message: str) -> None:
        for q in list(self._channels.get(channel, [])):
            await q.put(message)

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._channels.setdefault(channel, []).append(q)
        try:
            while True:
                yield await q.get()
        finally:
            subs = self._channels.get(channel) or []
            if q in subs:
                subs.remove(q)

    async def close(self) -> None:
        self._store.clear()
        self._counters.clear()
        self._channels.clear()
