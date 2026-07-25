"""Cognitive / semantic answer cache (A4 + vNext §3.6).

Same prompt + same options ⇒ (at low temperature) the same answer. Rather than
pay the latency/quota again, we hash the request and serve the stored text — a
TTL + LRU store keyed by a canonical hash of the messages and the answer-shaping
options (difficulty, temperature, token cap, model).

**vNext §3.6 (flag `advanced_rag.semantic_cache`, default OFF → byte-identical
to the exact-only cache below):**
  * **Per-user by construction (§10.2):** the key is scoped to the request user,
    so one account's answer can never serve another's.
  * **Context fingerprint:** an optional `context_fp` (attached files digest /
    active-artifact version / memory epoch) is part of the key — a changed
    context misses by construction.
  * **Freshness gate (§9.8 placeholder):** a VOLATILE prompt (time-sensitive
    cues) is never cache-served; the freshness classifier replaces the cue-list
    later.
  * **Near (embedding) tier (flag `semantic_cache_near`):** on an exact miss,
    embed the normalized prompt and serve a stored answer whose prompt is
    ≥ threshold cosine-similar within the SAME user+context scope — labeled
    "from a moment ago". Embeds in a worker thread, only when the per-scope
    index is non-empty. Fail-open.

Safety: only LOW-temperature calls are cached; only non-empty results stored.
Process-wide + thread-safe; survives across requests, not restarts.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

# Options that change the ANSWER (and so must be part of the key). Everything
# else (timeouts, avoid_model, session keys) is ignored.
_KEY_OPTS = ("difficulty", "temperature", "num_predict", "max_tokens", "format")
_DEFAULT_TEMP_CEILING = 0.5

_lock = threading.RLock()
_store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
# key -> ttl override (seconds), set by maybe_key for the SLOW freshness class,
# consumed once by put. Bounded implicitly by the store's own eviction.
_pending_ttl: dict[str, int] = {}
# Near (embedding) tier: scope -> list[(vec, key, ts)], bounded per scope.
_near_index: "dict[str, list]" = {}
_hits = 0
_misses = 0
_near_hits = 0

# Time-sensitive cues → never cache-served (a placeholder for §9.8 freshness).
_VOLATILE_CUES = (
    "today", "tonight", "tomorrow", "yesterday", "right now", "currently",
    "current ", "latest", "this week", "this month", "this year", "breaking",
    "news", "stock price", "share price", "weather", "who won", "as of",
    "up to date", "up-to-date", "real time", "real-time", "live score",
    "this morning", "this afternoon", "just released", "recently announced",
)
# Milder recency cues → SLOW class (cache, but with a shorter TTL).
_SLOW_CUES = ("recent", "nowadays", "these days", "modern", "this version")


def _flags() -> tuple[bool, int, int, float]:
    """(enabled, ttl_s, max_entries, temp_ceiling) from config, safe defaults."""
    try:
        from app.core.config_loader import cfg
        a = cfg.advanced_rag
        return (
            bool(getattr(a, "cognitive_cache", True)),
            int(getattr(a, "cognitive_cache_ttl_s", 3600)),
            int(getattr(a, "cognitive_cache_max", 512)),
            float(getattr(a, "cognitive_cache_temp_ceiling",
                          _DEFAULT_TEMP_CEILING)),
        )
    except Exception:  # noqa: BLE001
        return (True, 3600, 512, _DEFAULT_TEMP_CEILING)


def _semantic_flags() -> tuple[bool, int, bool, float, int]:
    """(semantic_v2_on, slow_ttl_s, near_on, near_threshold, near_max)."""
    try:
        from app.core.config_loader import cfg
        a = cfg.advanced_rag
        return (
            bool(getattr(a, "semantic_cache", False)),
            int(getattr(a, "semantic_cache_slow_ttl_s", 600)),
            bool(getattr(a, "semantic_cache_near", False)),
            float(getattr(a, "semantic_cache_near_threshold", 0.97)),
            int(getattr(a, "semantic_cache_near_max", 64)),
        )
    except Exception:  # noqa: BLE001
        return (False, 600, False, 0.97, 64)


def _scope() -> str:
    """The request user id (per-user cache §10.2), or 'anon'. Only consulted when
    the semantic-v2 flag is on, so the exact-only cache stays process-wide."""
    try:
        from storage.context import get_request_user_id
        return str(get_request_user_id() or "anon")
    except Exception:  # noqa: BLE001
        return "anon"


def freshness_class(text: str) -> str:
    """'volatile' | 'slow' | 'stable' from cheap lexical cues (a §9.8 stand-in)."""
    t = (text or "").lower()
    if any(cue in t for cue in _VOLATILE_CUES):
        return "volatile"
    if any(cue in t for cue in _SLOW_CUES):
        return "slow"
    return "stable"


def _canonical_messages(messages: list[dict]) -> list:
    """Reduce messages to (role, text) so formatting noise doesn't split keys.
    Image-bearing messages are NOT cacheable (the image isn't part of the key) —
    returns [] for those."""
    out = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            return []  # multimodal (OpenAI multipart) — don't cache
        if m.get("images"):
            return []  # vision turn (images side-channel) — don't cache
        out.append([str(m.get("role") or ""), str(content or "")])
    return out


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "") == "user":
            c = m.get("content")
            return str(c or "") if not isinstance(c, list) else ""
    return ""


def cache_key(messages: list[dict], options: dict, *,
              model: str | None = None, namespace: str = "",
              context_fp: str = "") -> str:
    msgs = _canonical_messages(messages)
    opts = {k: options.get(k) for k in _KEY_OPTS if k in (options or {})}
    payload: dict = {"ns": namespace, "model": model or "", "m": msgs, "o": opts}
    # v2: fold the per-user scope + context fingerprint into the key so an
    # answer is never reused across accounts or across a changed context.
    if _semantic_flags()[0]:
        payload["u"] = _scope()
        if context_fp:
            payload["ctx"] = context_fp
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def maybe_key(messages: list[dict], options: dict, *,
              model: str | None = None, namespace: str = "",
              context_fp: str = "") -> str | None:
    """A cache key IF this request is cacheable (enabled, low temp, text-only,
    and — under v2 — not VOLATILE), else None."""
    enabled, _ttl, _max, ceiling = _flags()
    if not enabled:
        return None
    temp = (options or {}).get("temperature")
    if temp is not None and float(temp) > ceiling:
        return None
    msgs = _canonical_messages(messages)
    if not msgs:
        return None  # empty or multimodal
    semantic_on, slow_ttl, *_ = _semantic_flags()
    if semantic_on:
        cls = freshness_class(_last_user_text(messages))
        if cls == "volatile":
            return None  # never cache a time-sensitive answer
        key = cache_key(messages, options or {}, model=model,
                        namespace=namespace, context_fp=context_fp)
        if cls == "slow":
            _pending_ttl[key] = slow_ttl   # consumed once by put()
        return key
    return cache_key(messages, options or {}, model=model, namespace=namespace,
                     context_fp=context_fp)


def relaxed_key(messages: list[dict], *, model: str | None = None,
                namespace: str = "", context_fp: str = "") -> str | None:
    """A prompt-ONLY key (ignores answer-shaping options) for the T5 exhaustion
    fallback (§2.1). Scoped per-user + context under v2, like the exact key."""
    msgs = _canonical_messages(messages)
    if not msgs:
        return None
    payload: dict = {"ns": namespace, "model": model or "", "m": msgs,
                     "relaxed": True}
    if _semantic_flags()[0]:
        payload["u"] = _scope()
        if context_fp:
            payload["ctx"] = context_fp
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def get(key: str | None) -> str | None:
    if not key:
        return None
    global _hits, _misses
    _, ttl, _max, _ = _flags()
    now = time.time()
    with _lock:
        entry = _store.get(key)
        if entry is None:
            _misses += 1
            return None
        expiry, value = entry
        if now >= expiry:
            _store.pop(key, None)
            _misses += 1
            return None
        _store.move_to_end(key)  # LRU touch
        _hits += 1
        return value


def put(key: str | None, value: str, *, ttl_s: int | None = None) -> None:
    if not key or not (value or "").strip():
        return
    _, ttl, max_entries, _ = _flags()
    with _lock:
        eff_ttl = ttl_s if ttl_s is not None else _pending_ttl.pop(key, ttl)
        _store[key] = (time.time() + eff_ttl, value)
        _store.move_to_end(key)
        while len(_store) > max_entries:
            _store.popitem(last=False)  # evict oldest


# ── near (embedding) tier ────────────────────────────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sa = sb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        sa += x * x
        sb += y * y
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return dot / ((sa ** 0.5) * (sb ** 0.5))


def _near_scope(model: str | None, context_fp: str) -> str:
    return f"{_scope()}|{model or ''}|{context_fp}"


def near_enabled() -> bool:
    """Whether the near (embedding) tier is on — so the caller (which owns the
    embedder, keeping this module free of an app.rag edge) knows to embed."""
    s = _semantic_flags()
    return bool(s[0] and s[2])


def normalized_prompt(messages: list[dict]) -> str:
    """The query text the near tier keys on — the caller uses this to embed the
    SAME normalized string this module compares against."""
    return _last_user_text(messages).strip().lower()


def near_get(messages: list[dict], options: dict, *, query_vec: list[float] | None,
             model: str | None = None, context_fp: str = ""):
    """On an exact miss, serve a stored answer whose prompt is ≥ threshold
    cosine-similar within this user+context scope. `query_vec` is the caller-
    supplied embedding of `normalized_prompt(messages)` (the caller owns the
    embedder). Returns ``(value, meta)`` or None. Fail-open, sync (the embed
    already happened off-thread in the caller)."""
    semantic_on, _slow, near_on, thresh, _max = _semantic_flags()
    if not (semantic_on and near_on) or not query_vec:
        return None
    # Same cacheability gate as the exact tier (low temp, text-only, not volatile).
    if maybe_key(messages, options or {}, model=model,
                 context_fp=context_fp) is None:
        return None
    scope = _near_scope(model, context_fp)
    with _lock:
        candidates = list(_near_index.get(scope) or [])
    if not candidates:
        return None
    best_key, best_sim, best_ts = None, 0.0, 0.0
    for cvec, ckey, cts in candidates:
        sim = _cosine(query_vec, cvec)
        if sim > best_sim:
            best_key, best_sim, best_ts = ckey, sim, cts
    if best_key is None or best_sim < thresh:
        return None
    value = get(best_key)
    if value is None:
        return None
    global _near_hits
    with _lock:
        _near_hits += 1
    return value, {"near": True, "similarity": round(best_sim, 4),
                   "age_s": max(0, int(time.time() - best_ts))}


def near_index(messages: list[dict], *, key: str, query_vec: list[float] | None,
               model: str | None = None, context_fp: str = "") -> None:
    """Index the prompt→key mapping for the near tier (called right after put),
    using the caller-supplied `query_vec`. Bounded per scope. Fail-open, sync."""
    semantic_on, _slow, near_on, _thresh, near_max = _semantic_flags()
    if not (semantic_on and near_on) or not key or not query_vec:
        return
    scope = _near_scope(model, context_fp)
    now = time.time()
    with _lock:
        lst = _near_index.setdefault(scope, [])
        # Replace an existing entry for this key, else append.
        lst[:] = [e for e in lst if e[1] != key]
        lst.append((list(query_vec), key, now))
        if len(lst) > near_max:
            del lst[:len(lst) - near_max]  # drop oldest


def clear() -> None:
    global _hits, _misses, _near_hits
    with _lock:
        _store.clear()
        _near_index.clear()
        _pending_ttl.clear()
        _hits = _misses = _near_hits = 0


def stats() -> dict:
    with _lock:
        total = _hits + _misses
        return {
            "entries": len(_store),
            "hits": _hits,
            "misses": _misses,
            "near_hits": _near_hits,
            "near_scopes": len(_near_index),
            "hit_rate": round(_hits / total, 3) if total else 0.0,
        }


__all__ = ["cache_key", "maybe_key", "relaxed_key", "get", "put",
           "near_get", "near_index", "near_enabled", "normalized_prompt",
           "freshness_class", "clear", "stats"]
