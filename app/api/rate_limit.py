"""Tiny in-process sliding-window rate limiter (vNext §10.1c hardening).

Used to blunt brute-force / abuse on the auth endpoints (login, register,
forgot-password). In-memory + per-process — enough for a single pod; a
multi-replica deploy would back this with Redis, but the call sites stay the
same. Keyed by whatever the caller passes (typically client-ip + email).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

_buckets: dict[str, deque] = defaultdict(deque)
_MAX_KEYS = 50_000  # coarse backstop so a flood of unique keys can't grow forever


def check_rate(key: str, *, max_attempts: int, window_s: float,
               now: float | None = None) -> bool:
    """Record an attempt for ``key``; return True if it's ALLOWED (still under
    the limit within the trailing ``window_s``), False if it should be blocked."""
    t = now if now is not None else time.time()
    if len(_buckets) > _MAX_KEYS:
        _buckets.clear()  # crude reset under pathological load
    dq = _buckets[key]
    cutoff = t - window_s
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= max_attempts:
        return False
    dq.append(t)
    return True


def reset_for_tests() -> None:
    _buckets.clear()


__all__ = ["check_rate", "reset_for_tests"]
