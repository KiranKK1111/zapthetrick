"""Adaptive benchmarking windows (intelligent-model-routing R9).

Rolling per-model windows of recent (success, latency) that contribute a
score down-rank when a model temporarily degrades, recovering as the window
clears (R9.2). Reuses the same in-memory, bounded style as the router penalties
rather than a parallel persistent tracker (R9.3); consumes perceived-speed
latency measurements where the caller supplies them.

`downrank(model_key) -> float` is an ADDITIVE score addend (0 = healthy) the
router adds only when `cfg.routing.adaptive_benchmark` is on, so it's
byte-for-byte today's ranking when off (Property 8/9).
"""
from __future__ import annotations

from collections import deque

_WINDOW = 12                      # recent samples per model
_LATENCY_BASELINE_MS = 8000.0     # a turn slower than this counts as "slow"
_MAX_DOWNRANK = 40.0              # cap so a bad window can't fully exile a model

# model_key -> deque[(success: bool, latency_ms: float|None)]
_WINDOWS: dict[object, deque] = {}


def record_outcome(model_key, success: bool, latency_ms: float | None = None) -> None:
    """Append one outcome to the model's rolling window. Never raises."""
    try:
        if model_key is None:
            return
        w = _WINDOWS.get(model_key)
        if w is None:
            w = deque(maxlen=_WINDOW)
            _WINDOWS[model_key] = w
        w.append((bool(success), latency_ms))
    except Exception:  # noqa: BLE001
        pass


def health_factor(model_key) -> float:
    """0..1 recent health (1 = healthy). Blends recent success rate with a
    latency factor. Empty window → 1.0 (no opinion)."""
    try:
        w = _WINDOWS.get(model_key)
        if not w:
            return 1.0
        n = len(w)
        succ = sum(1 for s, _ in w if s) / n
        lats = [lat for _, lat in w if isinstance(lat, (int, float))]
        if lats:
            avg = sum(lats) / len(lats)
            lat_factor = max(0.0, min(1.0, _LATENCY_BASELINE_MS / max(avg, 1.0)))
        else:
            lat_factor = 1.0
        # Success dominates; latency is a secondary nudge.
        return max(0.0, min(1.0, 0.7 * succ + 0.3 * lat_factor))
    except Exception:  # noqa: BLE001
        return 1.0


def downrank(model_key) -> float:
    """Additive score penalty for a degraded recent window (0 = healthy)."""
    return _MAX_DOWNRANK * (1.0 - health_factor(model_key))


def reset() -> None:
    _WINDOWS.clear()


__all__ = ["record_outcome", "health_factor", "downrank", "reset"]
