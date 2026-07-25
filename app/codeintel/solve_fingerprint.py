"""Problem-fingerprint cache for Solve (vNext §3.2).

`sha256(normalized statement + language)` → an already-solved problem returns
instantly (with a fresh beautify pass by the caller). A coding problem's solution
is genuinely user-agnostic (Two Sum in Python is the same for everyone), so this
cache is process-wide and NOT per-user — cross-user reuse is the feature.

Self-contained TTL + LRU store (no cross-package edge): the answer cache
(`app.llm.cache`) lives behind the `llm` package, and reaching it from `codeintel`
would add a new import edge for a small, distinct concern. Bounded + ephemeral —
a speed win, never a source of truth.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict

_WS = re.compile(r"\s+")

_lock = threading.RLock()
_store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_TTL_S = 24 * 3600.0     # a solved problem stays fresh for a day
_MAX = 512


def normalize_statement(statement: str) -> str:
    """Collapse whitespace + lowercase so trivial transcription noise (spacing,
    case) doesn't split an otherwise-identical problem."""
    return _WS.sub(" ", (statement or "").strip().lower())


def fingerprint(statement: str, language: str | None) -> str:
    key = normalize_statement(statement) + "" + (language or "").strip().lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get(fp: str | None) -> str | None:
    if not fp:
        return None
    now = time.time()
    with _lock:
        entry = _store.get(fp)
        if entry is None:
            return None
        expiry, value = entry
        if now >= expiry:
            _store.pop(fp, None)
            return None
        _store.move_to_end(fp)
        return value


def put(fp: str | None, solution: str) -> None:
    if not fp or not (solution or "").strip():
        return
    with _lock:
        _store[fp] = (time.time() + _TTL_S, solution)
        _store.move_to_end(fp)
        while len(_store) > _MAX:
            _store.popitem(last=False)


def clear() -> None:
    with _lock:
        _store.clear()


__all__ = ["normalize_statement", "fingerprint", "get", "put", "clear"]
