"""Cognitive cache (A4) — reuse a prior completion for an identical request.

Same prompt + same options ⇒ (at low temperature) the same answer. Rather than
pay the latency/quota again, we hash the request and serve the stored text. This
is the "cognitive cache" from report_2 §P2-10: a TTL + LRU store keyed by a
canonical hash of the messages and the answer-shaping options (difficulty,
temperature, token cap, model).

Safety: only LOW-temperature calls are cached (creative/varied calls aren't
frozen), and only non-empty results are stored. The cache is process-wide and
thread-safe; it survives across requests but not restarts (intentional — it's a
speed/quota optimization, not a source of truth). Disabled via
`advanced_rag.cognitive_cache`.
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
_hits = 0
_misses = 0


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


def _canonical_messages(messages: list[dict]) -> list:
    """Reduce messages to (role, text) so formatting noise doesn't split keys.
    Image-bearing messages are NOT cacheable (the image isn't part of the key,
    so caching would reuse one image's answer for another) — detected via a
    multipart `content` list OR a separate `images` key (this app's vision
    convention). Returns [] (→ not cacheable) for those."""
    out = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            return []  # multimodal (OpenAI multipart) — don't cache
        if m.get("images"):
            return []  # vision turn (images side-channel) — don't cache
        out.append([str(m.get("role") or ""), str(content or "")])
    return out


def cache_key(messages: list[dict], options: dict, *,
              model: str | None = None, namespace: str = "") -> str:
    msgs = _canonical_messages(messages)
    opts = {k: options.get(k) for k in _KEY_OPTS if k in (options or {})}
    payload = json.dumps(
        {"ns": namespace, "model": model or "", "m": msgs, "o": opts},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def maybe_key(messages: list[dict], options: dict, *,
              model: str | None = None, namespace: str = "") -> str | None:
    """A cache key IF this request is cacheable (enabled, low temp, text-only),
    else None."""
    enabled, _ttl, _max, ceiling = _flags()
    if not enabled:
        return None
    temp = (options or {}).get("temperature")
    if temp is not None and float(temp) > ceiling:
        return None
    msgs = _canonical_messages(messages)
    if not msgs:
        return None  # empty or multimodal
    return cache_key(messages, options or {}, model=model, namespace=namespace)


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


def put(key: str | None, value: str) -> None:
    if not key or not (value or "").strip():
        return
    _, ttl, max_entries, _ = _flags()
    with _lock:
        _store[key] = (time.time() + ttl, value)
        _store.move_to_end(key)
        while len(_store) > max_entries:
            _store.popitem(last=False)  # evict oldest


def clear() -> None:
    global _hits, _misses
    with _lock:
        _store.clear()
        _hits = _misses = 0


def stats() -> dict:
    with _lock:
        total = _hits + _misses
        return {
            "entries": len(_store),
            "hits": _hits,
            "misses": _misses,
            "hit_rate": round(_hits / total, 3) if total else 0.0,
        }


__all__ = ["cache_key", "maybe_key", "get", "put", "clear", "stats"]
