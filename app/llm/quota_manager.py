"""Free-tier quota & provider-rotation manager (roadmap Phase 5 #16).

The router already reacts to 429s (penalty + cooldown + sliding rate-limit
windows). This adds the PROACTIVE half the roadmap asks for: a per-provider
quota LEDGER with reset WINDOWS (daily / monthly free-tier caps), so providers
rotate as their free quota drains — before the 429 — and the system degrades to
local models when everything is exhausted.

Deterministic + injectable clock (tests advance time), in-process, fail-open.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

DAY = 86_400.0
MONTH = 30 * DAY
MINUTE = 60.0

# Sensible free-tier defaults for the zero-cost providers (requests per window).
# Conservative — the router's reactive 429 handling is the safety net; these just
# let us rotate PROACTIVELY. Override via configure().
DEFAULTS: dict[str, tuple[int, float]] = {
    "groq": (14_400, DAY),        # generous free daily
    "gemini": (1_500, DAY),       # free daily requests
    "openrouter": (1_000, DAY),   # free-model daily
    "cerebras": (14_400, DAY),
}


@dataclass
class QuotaWindow:
    limit: int          # requests per window (0 = unlimited / unknown)
    window_s: float
    used: int = 0
    window_start: float = field(default=0.0)


class QuotaManager:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._q: dict[str, QuotaWindow] = {}
        self._now = now or time.time
        for prov, (limit, win) in DEFAULTS.items():
            self._q[prov] = QuotaWindow(limit=limit, window_s=win,
                                        window_start=self._now())

    # ── config ───────────────────────────────────────────────────────────
    def configure(self, provider: str, *, limit: int, window_s: float) -> None:
        self._q[provider] = QuotaWindow(limit=limit, window_s=window_s,
                                        window_start=self._now())

    def _roll(self, w: QuotaWindow) -> None:
        """Reset the counter when the window has elapsed."""
        now = self._now()
        if w.window_s > 0 and now - w.window_start >= w.window_s:
            # Advance by whole windows so a long gap doesn't leave it stale.
            elapsed = now - w.window_start
            w.window_start += (int(elapsed // w.window_s)) * w.window_s
            w.used = 0

    # ── record + query ───────────────────────────────────────────────────
    def record(self, provider: str, n: int = 1) -> None:
        w = self._q.get(provider)
        if w is None:
            return
        self._roll(w)
        w.used += n

    def headroom(self, provider: str) -> int | None:
        """Remaining requests this window; None when unlimited/unknown."""
        w = self._q.get(provider)
        if w is None or w.limit <= 0:
            return None
        self._roll(w)
        return max(0, w.limit - w.used)

    def exhausted(self, provider: str) -> bool:
        h = self.headroom(provider)
        return h is not None and h <= 0

    def next_reset(self, provider: str) -> float | None:
        """Absolute epoch time when this provider's window resets."""
        w = self._q.get(provider)
        if w is None or w.window_s <= 0:
            return None
        self._roll(w)
        return w.window_start + w.window_s

    def rank(self, providers: list[str]) -> list[str]:
        """Order providers best-first: unlimited/unknown, then most headroom;
        exhausted providers sink to the back."""
        def key(p: str) -> tuple[int, float]:
            h = self.headroom(p)
            if h is None:
                return (0, -1.0)         # unknown/unlimited first
            if h <= 0:
                return (2, 0.0)          # exhausted last
            return (1, -float(h))        # more headroom = earlier
        return sorted(providers, key=key)

    def snapshot(self) -> list[dict]:
        out = []
        for prov, w in self._q.items():
            self._roll(w)
            out.append({
                "provider": prov,
                "limit": w.limit,
                "used": w.used,
                "headroom": (max(0, w.limit - w.used) if w.limit > 0 else None),
                "resets_at": (w.window_start + w.window_s if w.window_s > 0 else None),
            })
        return out


_manager = QuotaManager()


def quota_manager() -> QuotaManager:
    return _manager


__all__ = ["QuotaWindow", "QuotaManager", "quota_manager", "DAY", "MONTH", "MINUTE"]
