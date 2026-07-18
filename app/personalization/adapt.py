"""Response adaptation bridge (personalization-and-governance R2).

Maps the `UserModel` to a depth PREFERENCE consumed by the existing answer-depth
mechanic (`perceived-speed` R18 / `advanced-intent-reasoning`) — this spec only
SUPPLIES the signal, it does not re-implement depth (R2.1). It biases concise vs
detailed by the user's verbosity/expertise (R2.2) and NEVER overrides an explicit
per-turn depth choice (R2.3, Property 2). Pure; returns None when neutral.
"""
from __future__ import annotations

from app.personalization.user_model import UserModel, UNKNOWN

_VERBOSITY_TO_DEPTH = {
    "concise": "tldr",
    "balanced": "standard",
    "detailed": "deeper",
}


def preferred_depth(model: UserModel, explicit: str | None = None) -> str | None:
    """Return the depth label the answer mechanic should default to, or None for
    no bias. An explicit per-turn choice always wins (R2.3)."""
    try:
        if explicit:                       # explicit per-turn choice wins
            return explicit
        if model is None:
            return None
        if model.verbosity_pref in _VERBOSITY_TO_DEPTH:
            return _VERBOSITY_TO_DEPTH[model.verbosity_pref]
        # No explicit verbosity → infer from expertise (senior/expert lean terse).
        if model.expertise in ("senior", "expert"):
            return "tldr"
        if model.expertise == "beginner":
            return "deeper"
        return None
    except Exception:  # noqa: BLE001
        return None


def adapt_signals(model: UserModel) -> dict:
    """Additive signals the answer/clarifier path may consult (never overrides)."""
    try:
        if model is None or model.is_neutral:
            return {}
        out: dict = {}
        d = preferred_depth(model)
        if d:
            out["preferred_depth"] = d
        if model.comm_style != UNKNOWN:
            out["comm_style"] = model.comm_style
        if model.expertise != UNKNOWN:
            out["expertise"] = model.expertise
        return out
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["preferred_depth", "adapt_signals"]
