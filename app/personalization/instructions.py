"""User-authored custom instructions (Architecture §17).

Distinct from *learned* memory: these are standing directions the user wrote
themselves (tone, format, role, always/never) and expects applied every turn.
They are TRUSTED operator input — injected into the system prompt, not framed as
untrusted data — but sit BELOW the safety/trust boundary in the strict precedence:

    safety/trust boundary  ▷  user instructions  ▷  learned memory  ▷  intent defaults

So a user instruction can shape tone/format but can never override a safety rule.
Stored in `User.preferences['custom_instructions']`; capped so a pathological
paste can't blow up the prompt. All helpers are pure and fail-open.
"""
from __future__ import annotations

_KEY = "custom_instructions"
# Hard cap on stored/injected length — keeps the always-on prompt bounded.
MAX_CHARS = 2000


def load_custom_instructions(preferences: dict | None) -> str:
    """The user's stored custom instructions (trimmed + capped), or ""."""
    try:
        raw = (preferences or {}).get(_KEY)
        if not isinstance(raw, str):
            return ""
        return raw.strip()[:MAX_CHARS]
    except Exception:  # noqa: BLE001 — never break a turn over prefs
        return ""


def set_custom_instructions(preferences: dict | None, text: str | None) -> dict:
    """Return a NEW preferences dict with the custom instructions set (or cleared
    when text is blank). Trims + caps. Never mutates the input."""
    prefs = dict(preferences or {})
    cleaned = (text or "").strip()[:MAX_CHARS]
    if cleaned:
        prefs[_KEY] = cleaned
    else:
        prefs.pop(_KEY, None)
    return prefs


def frame_instructions(text: str | None) -> str:
    """Format the user's instructions as a TRUSTED, clearly-labelled system-prompt
    block that states its precedence. Returns "" when there's nothing to inject,
    so a user who set none gets today's prompt byte-for-byte."""
    t = (text or "").strip()[:MAX_CHARS]
    if not t:
        return ""
    return (
        "The user has provided the following standing instructions for how you "
        "should respond. Follow them for every reply UNLESS they conflict with "
        "the safety rules above (which always win) or with an explicit request "
        "in the current message. These are the user's own instructions — treat "
        "them as trusted:\n" + t
    )


def enabled() -> bool:
    """Master switch (`personalization.custom_instructions`). The real gate is
    whether the user actually set any text — an empty value changes nothing."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.personalization, "custom_instructions", True))
    except Exception:  # noqa: BLE001 — fail-open to on (presence still gates)
        return True


__all__ = [
    "load_custom_instructions", "set_custom_instructions",
    "frame_instructions", "enabled", "MAX_CHARS",
]
