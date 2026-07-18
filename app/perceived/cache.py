"""Predictive answer cache (perceived-speed R3).

Caches high-quality answers keyed by the request's normalized semantic content
so a repeated/likely-next request returns instantly. Backed by Redis/Dragonfly
when the workspace cache driver is configured, else a bounded in-process LRU.

A cached candidate is ALWAYS revalidated against the current context before it
is served (R3.3), and precompute of likely-next answers is gated by the
SpeculationBudget so it only runs when speculation is enabled and within budget
(R3.1/R3.2). The cache is bounded (R3.4).
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


def _normalize(prompt: str) -> str:
    """Whitespace/case-normalize so trivially-different phrasings share a key."""
    return " ".join((prompt or "").lower().split())


def cache_key(prompt: str, scope: str = "") -> str:
    """Exact key = sha256(normalized prompt + scope). Scope keeps one user's /
    workspace's answers from being served to another (privacy seam for R21)."""
    raw = f"{_normalize(prompt)}\x00{scope}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class _MemoryBackend:
    """Bounded in-process LRU (OrderedDict). Default when no Redis is configured."""

    def __init__(self, max_entries: int) -> None:
        self._d: "OrderedDict[str, str]" = OrderedDict()
        self._max = max(1, int(max_entries))

    async def get(self, key: str) -> str | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    async def put(self, key: str, value: str) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self._max:        # LRU eviction (R3.4)
            self._d.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._d.pop(key, None)

    def __len__(self) -> int:                  # for tests
        return len(self._d)


class _RedisBackend:
    """Redis/Dragonfly backend (best-effort). Keys are namespaced + TTL'd."""

    def __init__(self, client, ttl_seconds: int = 86_400) -> None:
        self._r = client
        self._ttl = ttl_seconds

    async def get(self, key: str) -> str | None:
        try:
            v = await self._r.get(f"pc:{key}")
            return v.decode() if isinstance(v, (bytes, bytearray)) else v
        except Exception:  # noqa: BLE001 — cache miss on any backend hiccup
            return None

    async def put(self, key: str, value: str) -> None:
        try:
            await self._r.set(f"pc:{key}", value, ex=self._ttl)
        except Exception:  # noqa: BLE001
            pass

    async def delete(self, key: str) -> None:
        try:
            await self._r.delete(f"pc:{key}")
        except Exception:  # noqa: BLE001
            pass


class PerceivedCache:
    """Exact-key predictive cache with revalidate-before-serve + bounded LRU."""

    def __init__(self, backend=None, max_entries: int | None = None) -> None:
        if max_entries is None:
            try:
                from app.core.config_loader import cfg
                max_entries = int(getattr(cfg.perceived,
                                          "predictive_cache_max_entries", 256))
            except Exception:  # noqa: BLE001
                max_entries = 256
        self._backend = backend if backend is not None else _MemoryBackend(max_entries)

    async def get(self, prompt: str, scope: str = "") -> str | None:
        return await self._backend.get(cache_key(prompt, scope))

    async def put(self, prompt: str, answer: str, scope: str = "") -> None:
        if answer:
            await self._backend.put(cache_key(prompt, scope), answer)

    async def serve_if_valid(
        self,
        prompt: str,
        scope: str = "",
        validate: Callable[[str], bool] | None = None,
    ) -> str | None:
        """Return the cached answer ONLY after `validate(cached)` passes (R3.3);
        an invalid entry is discarded and None returned so a fresh answer is
        generated."""
        cached = await self.get(prompt, scope)
        if cached is None:
            return None
        ok = True if validate is None else bool(validate(cached))
        if not ok:
            await self._backend.delete(cache_key(prompt, scope))
            return None
        return cached

    async def precompute_likely_next(
        self,
        candidates: list[str],
        generate: Callable[[str], Awaitable[str]],
        scope: str = "",
    ) -> int:
        """Budget-gated precompute of likely-next requests (R3.1/R3.2). Only
        runs while speculation is enabled + within budget; returns how many were
        precomputed. The actual generation is supplied by the caller."""
        from app.perceived.budget import budget

        done = 0
        for cand in candidates:
            if not budget.allow(kind="precompute"):
                break
            if await self.get(cand, scope) is not None:
                continue
            budget.account(1)
            try:
                ans = await generate(cand)
            except Exception:  # noqa: BLE001 — precompute must never raise
                continue
            if ans:
                await self.put(cand, ans, scope)
                done += 1
        return done


def _build_backend(max_entries: int):
    """Redis/Dragonfly when configured + reachable, else in-process LRU."""
    try:
        from app.core.config_loader import cfg
        cache_cfg = getattr(getattr(cfg, "database", None), "cache", None)
        backend = getattr(cache_cfg, "backend", "memory")
        url = getattr(cache_cfg, "url", "")
        if backend in ("redis", "dragonfly") and url:
            import redis.asyncio as redis  # type: ignore
            client = redis.from_url(url, socket_connect_timeout=2)
            return _RedisBackend(client)
    except Exception as exc:  # noqa: BLE001 — fall back to memory on any issue
        log.info("PerceivedCache: Redis unavailable (%s) — in-process LRU", exc)
    return _MemoryBackend(max_entries)


def default_cache() -> PerceivedCache:
    try:
        from app.core.config_loader import cfg
        mx = int(getattr(cfg.perceived, "predictive_cache_max_entries", 256))
    except Exception:  # noqa: BLE001
        mx = 256
    return PerceivedCache(backend=_build_backend(mx), max_entries=mx)


# ── Answer reuse cache (R14, R21) ────────────────────────────────────────────
import math
from collections import OrderedDict as _OD
from dataclasses import dataclass, field
from typing import Callable


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _AnswerEntry:
    key: str
    scope: str
    answer: str
    embedding: list[float] | None = None
    created: float = 0.0


class AnswerCache:
    """Reuse tier (R14) layered on the same key/scope/LRU machinery as the
    predictive cache. Stores high-quality completed answers, retrieves by exact
    key or semantic similarity (local embedder), revalidates before serving, and
    is strictly per-user scoped with `clear_user` for data-clear (R21)."""

    def __init__(self, max_entries: int | None = None, similarity: float | None = None) -> None:
        try:
            from app.core.config_loader import cfg
            max_entries = max_entries or int(
                getattr(cfg.perceived, "predictive_cache_max_entries", 256))
            similarity = similarity if similarity is not None else float(
                getattr(cfg.perceived, "cache_similarity_threshold", 0.95))
        except Exception:  # noqa: BLE001
            max_entries = max_entries or 256
            similarity = 0.95 if similarity is None else similarity
        self._max = max(1, int(max_entries))
        self._threshold = float(similarity)
        self._entries: "_OD[str, _AnswerEntry]" = _OD()
        self._by_scope: dict[str, set] = {}

    # ---- writes ----------------------------------------------------------
    def store(self, scope: str, prompt: str, answer: str, *,
              quality_ok: bool = True, embedding: list[float] | None = None) -> None:
        """Store only a high-quality completed answer (R14.1)."""
        if not answer or not quality_ok:
            return
        import time as _t
        key = cache_key(prompt, scope)
        self._entries[key] = _AnswerEntry(key, scope, answer, embedding, _t.monotonic())
        self._entries.move_to_end(key)
        self._by_scope.setdefault(scope, set()).add(key)
        while len(self._entries) > self._max:        # LRU bound (R21/R3.4)
            old_key, old = self._entries.popitem(last=False)
            self._by_scope.get(old.scope, set()).discard(old_key)

    # ---- reads -----------------------------------------------------------
    def get_exact(self, scope: str, prompt: str) -> str | None:
        e = self._entries.get(cache_key(prompt, scope))
        if e is None:
            return None
        self._entries.move_to_end(e.key)
        return e.answer

    def semantic_get(self, scope: str, prompt: str,
                     embed_fn: Callable[[str], list[float]],
                     threshold: float | None = None) -> str | None:
        """Best same-scope entry with cosine ≥ threshold (R14.2). Scope-isolated
        so one user's answer is never matched for another (R21.1/R21.2)."""
        thr = self._threshold if threshold is None else threshold
        try:
            q = embed_fn(prompt)
        except Exception:  # noqa: BLE001
            return None
        best, best_sim = None, thr
        for key in self._by_scope.get(scope, set()):
            e = self._entries.get(key)
            if e is None or e.embedding is None:
                continue
            sim = _cosine(q, e.embedding)
            if sim >= best_sim:
                best, best_sim = e, sim
        if best is not None:
            self._entries.move_to_end(best.key)
            return best.answer
        return None

    def serve(self, scope: str, prompt: str, *,
              validate: Callable[[str], bool] | None = None,
              embed_fn: Callable[[str], list[float]] | None = None,
              threshold: float | None = None) -> str | None:
        """Exact, then semantic; revalidate before serving (R14.3); discard +
        return None on validation failure so a fresh answer is generated
        (R21.4)."""
        ans = self.get_exact(scope, prompt)
        if ans is None and embed_fn is not None:
            ans = self.semantic_get(scope, prompt, embed_fn, threshold)
        if ans is None:
            return None
        if validate is not None and not validate(ans):
            self._invalidate(scope, prompt)        # stale/inconsistent (R14.4)
            return None
        return ans

    # ---- maintenance -----------------------------------------------------
    def _invalidate(self, scope: str, prompt: str) -> None:
        key = cache_key(prompt, scope)
        self._entries.pop(key, None)
        self._by_scope.get(scope, set()).discard(key)

    def clear_user(self, scope: str) -> int:
        """Delete every entry for a user/scope (data-clear — R21.3)."""
        keys = list(self._by_scope.pop(scope, set()))
        for k in keys:
            self._entries.pop(k, None)
        return len(keys)

    def __len__(self) -> int:
        return len(self._entries)


# Process-wide answer reuse cache (R14). One bounded LRU shared across requests
# so a question answered in one turn/conversation can be served instantly in a
# later one (same user scope). Lazily built; gated by `cfg.perceived.answer_cache`
# at the call sites — this singleton itself is inert until something stores.
_ANSWER_CACHE: "AnswerCache | None" = None


def answer_cache() -> "AnswerCache":
    global _ANSWER_CACHE
    if _ANSWER_CACHE is None:
        _ANSWER_CACHE = AnswerCache()
    return _ANSWER_CACHE


__all__ = ["PerceivedCache", "cache_key", "default_cache", "AnswerCache",
           "answer_cache"]
