"""Graceful degradation / self-healing (evaluation-and-reliability R6).

`guard(call, fallback, name)` runs a non-critical subsystem call (retrieval,
memory, a tool, reranking) and, on failure, returns a safe `fallback` while
recording a degradation event (R6.1/R6.3) so the turn still completes. Provider
failures are NOT handled here — they delegate to the existing router fallback
(R6.2). The guard NEVER wraps or bypasses a safety / destructive-action
confirmation (R6.4, Property 7).

Sync and async variants are provided. Degradation events are kept in a small,
bounded, process-wide ring so they can be surfaced as optional `degraded` meta.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Awaitable, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# Process-wide bounded ring of recent degradation events (observability, R6.3).
_EVENTS: "deque[dict]" = deque(maxlen=200)
_SEQ = 0


def record_event(name: str, error: str) -> None:
    global _SEQ
    _SEQ += 1
    _EVENTS.append({"seq": _SEQ, "subsystem": name, "error": error[:200]})


def recent_events(n: int = 20) -> list[dict]:
    return list(_EVENTS)[-n:]


def snapshot() -> int:
    """The current event sequence — capture at the start of a turn."""
    return _SEQ


def since(seq: int) -> list[dict]:
    """Degradation events recorded after `seq` (this turn's events)."""
    return [e for e in _EVENTS if e.get("seq", 0) > seq]


def reset_events() -> None:
    global _SEQ
    _EVENTS.clear()
    _SEQ = 0


def guard(call: Callable[[], T], fallback: T, name: str) -> T:
    """Run `call()`; on ANY exception return `fallback` + record a degradation
    event. Use only for NON-critical subsystems (never the safety guards)."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        record_event(name, f"{type(exc).__name__}: {exc}")
        log.info("degraded %s → fallback (%s)", name, exc)
        return fallback


async def guard_async(call: Callable[[], Awaitable[T]], fallback: T,
                      name: str) -> T:
    """Async variant of `guard`."""
    try:
        return await call()
    except Exception as exc:  # noqa: BLE001
        record_event(name, f"{type(exc).__name__}: {exc}")
        log.info("degraded %s → fallback (%s)", name, exc)
        return fallback


# Subsystem names that must NEVER be passed through the degradation guard —
# they own blocking safety / destructive-action decisions and must surface
# their own outcome, not a silent fallback (R6.4).
_PROTECTED = frozenset((
    "safety", "content_safety", "destructive_action", "destructive",
    "confirmation", "guardrail",
))


def is_protected(name: str) -> bool:
    """True when `name` is a safety/destructive-action guard that must not be
    degraded. Callers assert this before wrapping (defense in depth)."""
    return (name or "").strip().lower() in _PROTECTED


def safe_guard(call: Callable[[], T], fallback: T, name: str) -> T:
    """`guard` that refuses to wrap a protected safety subsystem — it re-raises
    so the safety outcome is never swallowed (R6.4)."""
    if is_protected(name):
        return call()    # run unguarded → its own (possibly blocking) outcome
    return guard(call, fallback, name)


__all__ = [
    "guard", "guard_async", "safe_guard", "is_protected",
    "record_event", "recent_events", "reset_events", "snapshot", "since",
]
