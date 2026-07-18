"""(model_id, text_hash) → token_count cache.

Architecture.md §3 targets >99% cache hit on token counts so we save
~2ms per LLM call. Bounded LRU keeps memory predictable.
"""
from __future__ import annotations

from collections import OrderedDict


class TokenCountCache:
    def __init__(self, *, max_entries: int = 10_000) -> None:
        self._cache: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._max = max_entries

    def get(self, model_id: str, text_hash: str) -> int | None:
        key = (model_id, text_hash)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, model_id: str, text_hash: str, count: int) -> None:
        key = (model_id, text_hash)
        self._cache[key] = count
        self._cache.move_to_end(key)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
