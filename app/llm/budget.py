"""Paid-usage budget guard for the hybrid strong tier (Phase P2-1).

When `routing.strong_tier_for_hard` is on, hard/expert turns may use a NON-free
(paid) model. This module caps how many paid requests are spent per calendar
month so the hybrid track can't run away with cost — over the cap, the router
falls back to free-only.

Soft, in-memory counter keyed by UTC month (resets on process restart — a
conservative cap is the goal, not exact billing). `cap <= 0` means unlimited.
"""
from __future__ import annotations

from datetime import datetime, timezone

# "YYYY-MM" -> paid request count
_paid: dict[str, int] = {}


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def record_paid(n: int = 1) -> None:
    """Record `n` paid (non-free) requests for the current month."""
    if n <= 0:
        return
    m = _month()
    _paid[m] = _paid.get(m, 0) + n


def paid_this_month() -> int:
    return _paid.get(_month(), 0)


def can_use_paid(cap: int) -> bool:
    """True if a paid request is still within the monthly cap (0 = unlimited)."""
    if cap is None or cap <= 0:
        return True
    return paid_this_month() < cap


def reset() -> None:
    """Clear the counter (tests / manual reset)."""
    _paid.clear()


__all__ = ["record_paid", "paid_this_month", "can_use_paid", "reset"]
