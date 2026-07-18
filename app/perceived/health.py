"""Provider health windows + latency-aware ranking (perceived-speed R9).

The existing `app/llm/router.py` already does difficulty-aware scoring, a
capability floor, 429 penalties, rate-limit headroom, and retryable fallback —
so most of R9 is already satisfied. The new piece is a rolling **health window**
per model (recent latency p50, error rate, availability) updated from each
request outcome (R9.5), plus a pure `LatencyRouter.select` that demonstrates the
difficulty×health ranking the live router can consume.

In-memory + fail-open: with no samples, health is neutral and ranking matches
today's difficulty-only behavior.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Difficulty → (intelligence_weight, speed_weight); mirrors router._DIFFICULTY.
_DIFFICULTY = {
    "trivial": (0.0, 4.0),
    "standard": (1.0, 2.5),
    "hard": (4.0, 0.3),
    "expert": (8.0, 0.0),
}
_W_HEALTH = 20.0   # how hard an unhealthy model is down-ranked


@dataclass
class _Window:
    latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    ok: int = 0
    err: int = 0
    last: float = 0.0
    # Circuit-breaker state: consecutive HARD failures (provider errors /
    # timeouts, NOT 429s) and the monotonic time the breaker last opened.
    consec_fail: int = 0
    opened_at: float = 0.0


class ProviderHealth:
    """Rolling per-key (model id or platform) health signals."""

    def __init__(self) -> None:
        self._w: dict = defaultdict(_Window)

    def record(self, key, *, latency_s: float | None = None, ok: bool = True,
               hard_failure: bool = False) -> None:
        """Record one request outcome.

        `hard_failure` marks a provider error / timeout (feeds the circuit
        breaker); a rate-limit 429 is NOT a hard failure — it has its own
        cooldown, so pass `ok=False` without `hard_failure` for those (or don't
        record them here at all).
        """
        w = self._w[key]
        if latency_s is not None and latency_s >= 0:
            w.latencies.append(latency_s)
        now = time.monotonic()
        if ok:
            w.ok += 1
            w.consec_fail = 0     # a success closes the breaker
            w.opened_at = 0.0
        else:
            w.err += 1
            if hard_failure:
                w.consec_fail += 1
                w.opened_at = now  # (re)arm the cooldown on each hard failure
        w.last = now

    def is_open(self, key, threshold: int, cooldown_s: float) -> bool:
        """True while the breaker is OPEN — the model has hit `threshold`
        consecutive hard failures and is still inside `cooldown_s`. After the
        cooldown it returns False (half-open: one probe is allowed; a success
        closes it, another hard failure re-arms the cooldown). Fail-open: any
        error or no data → not open."""
        try:
            w = self._w.get(key)
            if not w or w.consec_fail < max(1, int(threshold)) or w.opened_at <= 0:
                return False
            return (time.monotonic() - w.opened_at) < float(cooldown_s)
        except Exception:  # noqa: BLE001
            return False

    def latency_factor(self, key, baseline_s: float = 8.0) -> float:
        """0..1 recent speed from observed p50 (1 = fast, →0 = at/over baseline).
        No samples → 1.0 (neutral, no ranking effect)."""
        try:
            p50 = self.latency_p50(key)
            if p50 is None:
                return 1.0
            return max(0.0, min(1.0, float(baseline_s) / max(p50, 0.001)))
        except Exception:  # noqa: BLE001
            return 1.0

    def latency_p50(self, key) -> float | None:
        w = self._w.get(key)
        if not w or not w.latencies:
            return None
        s = sorted(w.latencies)
        return s[len(s) // 2]

    def error_rate(self, key) -> float:
        w = self._w.get(key)
        if not w:
            return 0.0
        tot = w.ok + w.err
        return (w.err / tot) if tot else 0.0

    def available(self, key) -> bool:
        """A model that is mostly failing is treated as unavailable."""
        return self.error_rate(key) < 0.8

    def health_score(self, key) -> float:
        """0..1, higher = healthier (low error + low latency). Neutral (1.0)
        with no samples → no effect on ranking."""
        w = self._w.get(key)
        if not w or (w.ok + w.err) == 0:
            return 1.0
        er = self.error_rate(key)
        p50 = self.latency_p50(key)
        lat_pen = 0.0 if p50 is None else min(1.0, p50 / 10.0)
        return max(0.0, 1.0 - er - 0.3 * lat_pen)

    def reset(self) -> None:
        self._w.clear()


class LatencyRouter:
    """Pure difficulty×health ranking primitive (the live router consumes the
    same signals; this makes the policy testable)."""

    def __init__(self, health: ProviderHealth | None = None) -> None:
        self._health = health or ProviderHealth()

    def select(self, difficulty: str, candidates: list[dict]) -> list[dict]:
        """Return `candidates` ordered best-first for `difficulty`, down-ranking
        unhealthy/unavailable models. Each candidate: `{id, intelligence_rank,
        speed_rank}` (lower rank = stronger/faster)."""
        intel_w, speed_w = _DIFFICULTY.get(difficulty, _DIFFICULTY["standard"])

        def _score(c: dict) -> float:
            h = self._health.health_score(c.get("id"))
            unavailable_pen = 0.0 if self._health.available(c.get("id")) else 1e6
            return (
                intel_w * (c.get("intelligence_rank", 100) or 100)
                + speed_w * (c.get("speed_rank", 100) or 100)
                + _W_HEALTH * (1.0 - h)
                + unavailable_pen
            )

        return sorted(candidates, key=_score)


# Process-wide health tracker — the engine records outcomes here.
health = ProviderHealth()

__all__ = ["ProviderHealth", "LatencyRouter", "health"]
