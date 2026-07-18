"""Cache + pub/sub abstraction.

Three backends: dragonfly (default, Redis-compatible), redis (alt),
memory (in-process fallback for tests / no-docker dev).

All three speak the same [Cache] interface — TTL get/set + pub/sub.
"""
from .base import Cache
from .factory import get_cache, close_cache

__all__ = ["Cache", "get_cache", "close_cache"]
