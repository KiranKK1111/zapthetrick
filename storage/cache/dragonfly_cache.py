"""DragonflyDB adapter.

Wire-compatible with Redis, so we delegate to [RedisCache]. Kept as a
separate file so the factory can branch on `cfg.database.cache.backend`
without leaking provider names into [RedisCache]'s docstrings.
"""
from __future__ import annotations

from .redis_cache import RedisCache


class DragonflyCache(RedisCache):
    """Same client, different default endpoint. Inherits everything."""
