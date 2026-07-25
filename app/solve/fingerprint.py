"""Problem-fingerprint solution cache (vNext §3.2).

An already-solved coding problem — same problem, same target language — should
not be re-reasoned from scratch: we key the solution by a stable fingerprint of
its NORMALIZED statement + language, and on a repeat re-serve it instantly (a
cheap beautify pass is the caller's move). The fingerprint is deterministic and
whitespace/punctuation/case-insensitive so trivial edits (a reflowed paragraph,
a trailing period) still hit.

Store: a self-contained bounded LRU with revalidate-before-serve (the same
contract as Component D's §3.6 cache, kept local so `solve` takes no dependency
on `perceived`). Per-user isolation rides the `scope` string the caller passes
(the request user id) folded into the key. Pure key derivation + a thin async
wrapper; fully fail-open — any cache error just means "no hit, solve normally".
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict

# Collapse runs of anything that isn't a word char into a single space, so
# reflow/punctuation/case differences in the SAME problem still fingerprint the
# same. (We keep digits — "10^5" vs "10^9" are genuinely different constraints.)
_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize_statement(text: str) -> str:
    """Lowercase + collapse non-alphanumerics to single spaces + trim. Stable
    across cosmetic edits; never raises."""
    try:
        return _NON_WORD.sub(" ", (text or "").lower()).strip()
    except Exception:  # noqa: BLE001
        return ""


def fingerprint(statement: str, language: str) -> str:
    """`sha256(normalized_statement | language)` hex. A blank statement yields an
    empty fingerprint so the caller skips the cache entirely."""
    norm = normalize_statement(statement)
    if not norm:
        return ""
    lang = (language or "").strip().lower()
    return hashlib.sha256(f"{norm}\x00{lang}".encode()).hexdigest()


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.code_solver, "fingerprint_cache", False))
    except Exception:  # noqa: BLE001
        return False


class _LRU:
    """A tiny thread-safe bounded LRU (insertion-order eviction). Sync ops — the
    module's async wrappers keep the caller's `await` shape without an event
    loop dependency."""

    def __init__(self, cap: int = 256) -> None:
        self._cap = max(1, cap)
        self._d: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)      # mark most-recently-used
            return self._d[key]

    def put(self, key: str, val: str) -> None:
        with self._lock:
            self._d[key] = val
            self._d.move_to_end(key)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)  # evict least-recently-used

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


def _store() -> "_LRU":
    """The process-wide solution LRU (bounded by `code_solver.fingerprint_cache_max`)."""
    global _CACHE
    if _CACHE is None:
        cap = 256
        try:
            from app.core.config_loader import cfg
            cap = int(getattr(cfg.code_solver, "fingerprint_cache_max", 256))
        except Exception:  # noqa: BLE001
            cap = 256
        _CACHE = _LRU(cap)
    return _CACHE


_CACHE: "_LRU | None" = None


def _valid_solution(text: str) -> bool:
    """Never serve an error-marked / empty entry (mirrors the answer cache's
    guard) so a poisoned row can't be handed back as a solved problem."""
    if not text or not text.strip():
        return False
    return "[LLM error:" not in text and "[Persona could not" not in text


def _key(statement: str, language: str, scope: str) -> str:
    """Fingerprint folded with the caller's per-user scope so users never share
    solution rows."""
    fp = fingerprint(statement, language)
    return f"{scope or 'anon'}\x00{fp}" if fp else ""


async def get(statement: str, language: str, *, scope: str = "") -> str | None:
    """Return a previously-cached solution for this (statement, language), or
    None. Revalidates before serving (a poisoned row is dropped); fail-open."""
    if not enabled():
        return None
    key = _key(statement, language, scope)
    if not key:
        return None
    try:
        store = _store()
        cached = store.get(key)
        if cached is None:
            return None
        if not _valid_solution(cached):
            store.delete(key)
            return None
        return cached
    except Exception:  # noqa: BLE001
        return None


async def put(statement: str, language: str, solution: str,
              *, scope: str = "") -> None:
    """Cache a fresh solution under this (statement, language). No-op on a blank
    fingerprint / invalid solution / disabled; fail-open."""
    if not enabled() or not _valid_solution(solution):
        return
    key = _key(statement, language, scope)
    if not key:
        return
    try:
        _store().put(key, solution)
    except Exception:  # noqa: BLE001
        pass


def clear() -> None:
    """Test/maintenance hook — drop the process cache."""
    global _CACHE
    _CACHE = None


__all__ = ["normalize_statement", "fingerprint", "enabled", "get", "put",
           "clear"]
