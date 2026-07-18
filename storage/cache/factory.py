"""Build the configured [Cache] from `cfg.database.cache.backend`."""
from __future__ import annotations

import logging

from app.core.config_loader import cfg

from .base import Cache
from .dragonfly_cache import DragonflyCache
from .memory_cache import MemoryCache
from .redis_cache import RedisCache


log = logging.getLogger(__name__)
_singleton: Cache | None = None


def get_cache() -> Cache:
    global _singleton
    if _singleton is not None:
        return _singleton

    section = cfg.database.cache
    backend = (section.backend or "memory").lower()

    if backend == "dragonfly":
        _singleton = DragonflyCache(
            url=section.url, default_ttl_seconds=section.default_ttl_seconds
        )
    elif backend == "redis":
        _singleton = RedisCache(
            url=section.url, default_ttl_seconds=section.default_ttl_seconds
        )
    elif backend == "memory":
        _singleton = MemoryCache(default_ttl_seconds=section.default_ttl_seconds)
    else:
        log.warning("Unknown cache backend %r; falling back to memory.", backend)
        _singleton = MemoryCache(default_ttl_seconds=section.default_ttl_seconds)
    return _singleton


async def close_cache() -> None:
    global _singleton
    if _singleton is not None:
        await _singleton.close()
    _singleton = None
