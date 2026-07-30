"""User profile fields — the editable identity behind the Profile screen.

Before this, a user's identity was whatever they typed at registration: `User.name`
and nothing else. There was no preferred name, no avatar, and no way to change any
of it from inside the app (no `PATCH /me` existed). The Profile screen needs all
three, so this module owns them.

Storage: `User.name` stays the canonical **full name** (it is a real column, used
by auth), while the preferred name and avatar live in the `User.preferences`
JSONB blob — the same extension point `custom_instructions` already uses. That
avoids a migration for two optional presentation fields, and keeps
`preferences` as the one place per-user settings accumulate.

The avatar is stored as a **data URL**, capped hard. The client resizes to a small
square before upload, so this is tens of KB, not megabytes — deliberately chosen
over a blob-storage round trip for a field that is read on every profile load and
never streamed. The cap is enforced HERE (not just client-side) because the field
is user-supplied: an oversized or non-image payload is rejected rather than
persisted into a JSON column that every request loads.

Every function is pure over the preferences dict and never mutates its input,
matching `instructions.py`.
"""
from __future__ import annotations

import re

_DISPLAY_NAME = "display_name"
_AVATAR = "avatar"

# A preferred name is a label, not prose.
MAX_NAME_CHARS = 60
MAX_DISPLAY_NAME_CHARS = 40
# ~192 KB of base64 ≈ a 140 KB image. A 256px JPEG avatar lands far under this;
# the cap exists so a hand-rolled request can't bloat the row.
MAX_AVATAR_CHARS = 192 * 1024
# Only real raster image data URLs. `svg+xml` is deliberately EXCLUDED: an SVG is
# executable content, and this string is handed to an <img>/Image widget.
_AVATAR_RE = re.compile(r"^data:image/(png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=]+$")


def _clean(value, limit: int) -> str:
    """A trimmed, length-capped, single-line string. Control characters (a name
    field is not a place for newlines or NULs) are stripped."""
    if not isinstance(value, str):
        return ""
    text = "".join(ch for ch in value if ch.isprintable())
    return text.strip()[:limit]


def clean_full_name(value) -> str:
    """The canonical full name, for `User.name`."""
    return _clean(value, MAX_NAME_CHARS)


def clean_display_name(value) -> str:
    """What the assistant should call the user."""
    return _clean(value, MAX_DISPLAY_NAME_CHARS)


def clean_avatar(value) -> str:
    """A validated avatar data URL, or "" when absent/invalid/oversized.

    Returns "" rather than raising so a bad avatar never blocks saving the rest of
    the profile — the caller reports it, the name still saves.
    """
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    if len(raw) > MAX_AVATAR_CHARS:
        return ""
    if not _AVATAR_RE.match(raw):
        return ""
    return raw


def load_display_name(preferences: dict | None) -> str:
    try:
        return clean_display_name((preferences or {}).get(_DISPLAY_NAME))
    except Exception:  # noqa: BLE001 — never break a request over prefs
        return ""


def load_avatar(preferences: dict | None) -> str:
    try:
        return clean_avatar((preferences or {}).get(_AVATAR))
    except Exception:  # noqa: BLE001
        return ""


def set_display_name(preferences: dict | None, value) -> dict:
    """A NEW preferences dict with the preferred name set (or cleared when blank)."""
    prefs = dict(preferences or {})
    cleaned = clean_display_name(value)
    if cleaned:
        prefs[_DISPLAY_NAME] = cleaned
    else:
        prefs.pop(_DISPLAY_NAME, None)
    return prefs


def set_avatar(preferences: dict | None, value) -> dict:
    """A NEW preferences dict with the avatar set (or cleared when blank/invalid)."""
    prefs = dict(preferences or {})
    cleaned = clean_avatar(value)
    if cleaned:
        prefs[_AVATAR] = cleaned
    else:
        prefs.pop(_AVATAR, None)
    return prefs


def preferred_name(user_name: str | None, preferences: dict | None) -> str:
    """What to actually call the user: the preferred name, else the first word of
    the full name, else "". Used to address the user in generated text."""
    display = load_display_name(preferences)
    if display:
        return display
    full = clean_full_name(user_name)
    return full.split()[0] if full else ""


def frame_preferred_name(name: str | None) -> str:
    """The system-prompt block that tells the assistant what to call the user.

    A pure function, like `instructions.frame_instructions`, so the wording is
    testable without standing up an agent — and so the Profile field is actually
    USED rather than merely stored and displayed back.

    Returns "" when there is no name, and instructs restraint: a model told a name
    will otherwise open every reply with it.
    """
    called = clean_display_name(name)
    if not called:
        return ""
    return (f"The user's name is {called}. Address them as {called} where a name "
            f"reads naturally; do not open every reply with it.")


def profile_payload(user) -> dict:
    """The wire shape the Profile screen reads."""
    prefs = getattr(user, "preferences", None)
    return {
        "id": str(getattr(user, "id", "") or ""),
        "email": getattr(user, "email", None),
        "full_name": clean_full_name(getattr(user, "name", None)),
        "display_name": load_display_name(prefs),
        "avatar": load_avatar(prefs),
        "limits": {
            "full_name": MAX_NAME_CHARS,
            "display_name": MAX_DISPLAY_NAME_CHARS,
            "avatar": MAX_AVATAR_CHARS,
        },
    }


__all__ = [
    "MAX_NAME_CHARS", "MAX_DISPLAY_NAME_CHARS", "MAX_AVATAR_CHARS",
    "clean_full_name", "clean_display_name", "clean_avatar",
    "load_display_name", "load_avatar", "set_display_name", "set_avatar",
    "preferred_name", "frame_preferred_name", "profile_payload",
]
