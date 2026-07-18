"""Speculation budget + master kill-switch (perceived-speed R19).

All *Speculative_Work* — prefetch (R1), predictive caching (R3), and speculative
drafting (R4, Phase 2) — must pass through this budget so it can be globally
disabled, capped per period, and concurrency-limited. With
`cfg.perceived.speculation_enabled=False` (the default) `allow()` always returns
False, so no speculative path runs (R19.4).

`account()` records speculative units against the period budget (R19.5); the
underlying model calls are still routed + tracked normally by the LLM engine.
`cancel_scope()` tracks an in-flight speculative draft so the concurrency cap is
honored and is released promptly when the work completes or is superseded (R19.3).
"""
from __future__ import annotations

import time
from contextlib import contextmanager


def speculation_enabled() -> bool:
    """Master switch — `cfg.perceived.speculation_enabled`. Fail-closed: any
    error reading config means speculation is OFF (today's behavior)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "perceived", None),
                            "speculation_enabled", False))
    except Exception:  # noqa: BLE001
        return False


class SpeculationBudget:
    """Process-wide gate + accounting for all speculative work."""

    def __init__(self) -> None:
        self._period_start = time.monotonic()
        self._used = 0
        self._active_drafts = 0

    # ---- config helpers --------------------------------------------------
    @staticmethod
    def _cfg():
        try:
            from app.core.config_loader import cfg
            return getattr(cfg, "perceived", None)
        except Exception:  # noqa: BLE001
            return None

    def _roll_period(self) -> None:
        c = self._cfg()
        window = float(getattr(c, "speculation_period_seconds", 3600) or 3600)
        if time.monotonic() - self._period_start >= window:
            self._period_start = time.monotonic()
            self._used = 0

    # ---- gating ----------------------------------------------------------
    def allow(self, *, kind: str = "work") -> bool:
        """True when speculative work of `kind` may start now: speculation is
        enabled AND the period budget isn't exhausted AND (for drafts) the
        concurrency cap isn't reached."""
        if not speculation_enabled():
            return False
        c = self._cfg()
        self._roll_period()
        period_budget = int(getattr(c, "speculation_period_budget", 0) or 0)
        if period_budget > 0 and self._used >= period_budget:
            return False
        if kind == "draft":
            cap = int(getattr(c, "max_concurrent_drafts", 2) or 2)
            if self._active_drafts >= cap:
                return False
        return True

    # ---- accounting ------------------------------------------------------
    def account(self, n: int = 1) -> None:
        """Record `n` speculative units against the current period."""
        self._roll_period()
        self._used += max(0, n)

    @contextmanager
    def cancel_scope(self, *, kind: str = "draft"):
        """Track an in-flight speculative draft for the concurrency cap; the
        slot is released on exit (completion, supersession, or cancellation)."""
        self._active_drafts += 1
        try:
            yield
        finally:
            self._active_drafts = max(0, self._active_drafts - 1)

    @property
    def active_drafts(self) -> int:
        return self._active_drafts

    def reset(self) -> None:
        """Reset accounting (tests / new period)."""
        self._period_start = time.monotonic()
        self._used = 0
        self._active_drafts = 0


# Process-wide singleton — all speculative paths share it.
budget = SpeculationBudget()

__all__ = ["SpeculationBudget", "budget", "speculation_enabled"]
