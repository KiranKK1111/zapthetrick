"""User-controlled reasoning modes — Fast / Balanced / Thorough (P5 #26).

Depth is normally chosen automatically from difficulty (#7). This adds an
explicit user-facing LEVER the router and governor honour on top of that:

  * FAST      — bias toward a snappy, cheaper answer (shift routing one band
                DOWN, pick the governor's fast pipeline).
  * BALANCED  — today's automatic behaviour (no change).
  * THOROUGH  — bias toward a stronger model + the deep pipeline (shift routing
                one band UP).

The mode lives in a `ContextVar` so it's per-request and never leaks across
turns. `route_and_complete` / `route_and_stream` (engine.py) apply
`effective_difficulty` when routing; the governor consults `quality_budget`.
Default is BALANCED → byte-identical to today. Fail-open throughout.
"""
from __future__ import annotations

from contextvars import ContextVar

FAST = "fast"
BALANCED = "balanced"
THOROUGH = "thorough"
MODES = (FAST, BALANCED, THOROUGH)

# Ordered difficulty ladder used for band shifting.
_LADDER = ["trivial", "standard", "hard", "expert"]

_mode: ContextVar[str] = ContextVar("dtt_reasoning_mode", default=BALANCED)


def set_mode(mode: str | None) -> str:
    """Set the current reasoning mode (per request). Invalid → BALANCED."""
    m = (mode or "").strip().lower()
    if m not in MODES:
        m = BALANCED
    _mode.set(m)
    return m


def current_mode() -> str:
    try:
        return _mode.get()
    except Exception:  # noqa: BLE001
        return BALANCED


def reset() -> None:
    _mode.set(BALANCED)


def from_signals(difficulty: str | None = None, depth: str | None = None) -> str:
    """Derive a mode from the request's existing signals (the FE already sends
    `depth` and a `difficulty` override). depth tldr → FAST; exhaustive/deeper →
    THOROUGH; an explicit expert difficulty → THOROUGH; else BALANCED."""
    try:
        d = (depth or "").strip().lower()
        if d in ("tldr", "brief"):
            return FAST
        if d in ("deeper", "exhaustive"):
            return THOROUGH
        diff = (difficulty or "").strip().lower()
        if diff == "expert":
            return THOROUGH
        if diff == "trivial":
            return FAST
        return BALANCED
    except Exception:  # noqa: BLE001
        return BALANCED


def _shift(difficulty: str, steps: int) -> str:
    try:
        i = _LADDER.index((difficulty or "standard").lower())
    except ValueError:
        return difficulty
    j = max(0, min(len(_LADDER) - 1, i + steps))
    return _LADDER[j]


def effective_difficulty(difficulty: str | None, mode: str | None = None) -> str:
    """Apply the current (or given) mode to a difficulty band. FAST shifts one
    band down (faster/cheaper), THOROUGH one band up (stronger). BALANCED → the
    input unchanged. Fail-open to the input."""
    try:
        base = (difficulty or "standard").lower()
        m = (mode or current_mode())
        if m == FAST:
            return _shift(base, -1)
        if m == THOROUGH:
            return _shift(base, +1)
        return base
    except Exception:  # noqa: BLE001
        return difficulty or "standard"


def quality_budget(mode: str | None = None) -> str:
    """Map the mode to the governor's Budgets.quality ('fast'|'balanced'|'thorough')."""
    m = mode or current_mode()
    return m if m in MODES else BALANCED


__all__ = [
    "FAST", "BALANCED", "THOROUGH", "MODES",
    "set_mode", "current_mode", "reset", "from_signals",
    "effective_difficulty", "quality_budget",
]
