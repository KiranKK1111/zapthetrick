"""Idempotency registry for composed sends (vNext §3.4).

The client attaches a UUID to every composed message; the backend dedups on it so
a flaky-network retry, a reconnect replay, or a double-tap can never create a
duplicate turn or double-fire the pipeline. Deletes the entire "it answered
twice" bug class.

Process-wide, thread-safe, TTL-bounded. Deliberately in-memory: it guards against
near-simultaneous duplicates (the real failure mode), survives across requests,
and a restart simply forgets old keys (a post-restart resend is a new turn, which
is correct). Fail-open: a bad key is treated as "no key" (proceed normally).
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.RLock()
# key -> {"status": "in_flight"|"done", "result": dict|None, "ts": float}
_STORE: dict[str, dict] = {}
_TTL_S = 600.0          # forget a key 10 min after its last update
_MAX = 4096             # hard cap so a flood can't grow unbounded


def _gc(now: float) -> None:
    if len(_STORE) <= _MAX:
        for k in [k for k, e in _STORE.items() if now - e["ts"] > _TTL_S]:
            _STORE.pop(k, None)
        return
    # Over the cap → evict oldest first, then the TTL sweep.
    for k, _ in sorted(_STORE.items(), key=lambda kv: kv[1]["ts"])[:len(_STORE) - _MAX]:
        _STORE.pop(k, None)
    for k in [k for k, e in _STORE.items() if now - e["ts"] > _TTL_S]:
        _STORE.pop(k, None)


def claim(key: str | None) -> tuple[bool, dict | None]:
    """Try to claim ``key`` for a NEW turn.

    * ``(True, None)``  → the caller owns this key; proceed with the turn.
    * ``(False, result)`` → duplicate. ``result`` is the completed record when
      the first turn already finished, or ``None`` while it is still in flight
      (a double-tap arriving mid-stream).

    An empty/None key means "no idempotency requested" → always ``(True, None)``.
    """
    if not key:
        return True, None
    now = time.time()
    with _LOCK:
        _gc(now)
        e = _STORE.get(key)
        if e is None:
            _STORE[key] = {"status": "in_flight", "result": None, "ts": now}
            return True, None
        e["ts"] = now
        return False, (e["result"] if e["status"] == "done" else None)


def complete(key: str | None, result: dict) -> None:
    """Record a turn's terminal result so a later duplicate returns it."""
    if not key:
        return
    with _LOCK:
        _STORE[key] = {"status": "done", "result": dict(result or {}),
                       "ts": time.time()}


def release(key: str | None) -> None:
    """Release an IN-FLIGHT claim whose turn failed before completing, so a
    genuine retry can proceed. A completed key is left intact (its result is
    still the idempotent answer)."""
    if not key:
        return
    with _LOCK:
        e = _STORE.get(key)
        if e is not None and e["status"] == "in_flight":
            _STORE.pop(key, None)


def reset_for_tests() -> None:
    with _LOCK:
        _STORE.clear()


__all__ = ["claim", "complete", "release", "reset_for_tests"]
