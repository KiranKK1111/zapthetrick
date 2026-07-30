"""User profile fields (the Profile screen's backend).

Before this there was no profile at all: `User.name` from registration and
nothing else — no preferred name, no avatar, and no route to change either. These
tests pin the storage rules, the validation (the avatar is user-supplied and lands
in a JSON column every request loads), and the PATCH semantics that let a client
tell "leave this alone" apart from "clear this".
"""
from __future__ import annotations

import asyncio

import pytest

import app.personalization.profile as P

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class _User:
    """Minimal stand-in for the ORM row (the module only reads attributes)."""

    def __init__(self, name=None, preferences=None, uid="u1", email="a@b.c"):
        self.id = uid
        self.email = email
        self.name = name
        self.preferences = preferences


# ---- cleaning ------------------------------------------------------------
def test_names_are_trimmed_capped_and_single_line():
    assert P.clean_full_name("  Kiran Jana  ") == "Kiran Jana"
    # A name field is not a place for newlines or control characters.
    assert P.clean_display_name("Ki\nran") == "Kiran"
    assert P.clean_display_name("K\x00J") == "KJ"
    assert len(P.clean_full_name("x" * 500)) == P.MAX_NAME_CHARS
    assert len(P.clean_display_name("x" * 500)) == P.MAX_DISPLAY_NAME_CHARS


def test_non_strings_and_blanks_clean_to_empty():
    for bad in (None, 123, [], {}, "   "):
        assert P.clean_full_name(bad) == ""
        assert P.clean_display_name(bad) == ""
        assert P.clean_avatar(bad) == ""


# ---- avatar validation ---------------------------------------------------
def test_a_valid_raster_data_url_is_accepted():
    for mime in ("png", "jpeg", "jpg", "webp", "gif"):
        assert P.clean_avatar(f"data:image/{mime};base64,AAAA")


def test_svg_is_rejected():
    # An SVG is executable content and this string is handed to an <img>.
    assert P.clean_avatar("data:image/svg+xml;base64,AAAA") == ""


def test_non_image_and_malformed_payloads_are_rejected():
    for bad in (
        "https://example.com/a.png",
        "data:text/html;base64,AAAA",
        "data:image/png,notbase64",
        "data:image/png;base64,not base64!",
        "javascript:alert(1)",
        PNG.replace("base64", "base32"),
    ):
        assert P.clean_avatar(bad) == "", bad


def test_an_oversized_avatar_is_rejected_server_side():
    # The cap is enforced here, not just in the client — the field is
    # user-supplied and lives in a column every request loads.
    huge = "data:image/png;base64," + "A" * (P.MAX_AVATAR_CHARS + 10)
    assert P.clean_avatar(huge) == ""


# ---- preferences round-trip ---------------------------------------------
def test_set_and_load_display_name():
    prefs = P.set_display_name(None, "Kiran")
    assert P.load_display_name(prefs) == "Kiran"


def test_blank_clears_rather_than_storing_empty():
    prefs = P.set_display_name({"display_name": "Kiran"}, "  ")
    assert "display_name" not in prefs
    assert P.load_display_name(prefs) == ""


def test_an_invalid_avatar_clears_instead_of_persisting_junk():
    prefs = P.set_avatar({"avatar": PNG}, "data:image/svg+xml;base64,AAAA")
    assert "avatar" not in prefs


def test_setters_never_mutate_the_input():
    original = {"custom_instructions": "keep me"}
    snapshot = dict(original)
    P.set_display_name(original, "Kiran")
    P.set_avatar(original, PNG)
    assert original == snapshot


def test_other_preference_keys_survive():
    # `preferences` is shared with custom_instructions and anything added later.
    prefs = P.set_display_name({"custom_instructions": "be terse"}, "Kiran")
    prefs = P.set_avatar(prefs, PNG)
    assert prefs["custom_instructions"] == "be terse"
    assert prefs["display_name"] == "Kiran"
    assert prefs["avatar"] == PNG


def test_loaders_never_raise_on_a_corrupt_preferences_blob():
    for bad in (None, {}, {"display_name": 5}, {"avatar": ["x"]}):
        assert P.load_display_name(bad) == ""
        assert P.load_avatar(bad) == ""


# ---- preferred_name ----------------------------------------------------
def test_preferred_name_prefers_the_display_name():
    assert P.preferred_name("Kiran Jana", {"display_name": "KJ"}) == "KJ"


def test_preferred_name_falls_back_to_the_first_word_of_the_full_name():
    assert P.preferred_name("Kiran Jana", None) == "Kiran"
    assert P.preferred_name("Kiran", {}) == "Kiran"


def test_preferred_name_is_empty_when_nothing_is_known():
    assert P.preferred_name(None, None) == ""
    assert P.preferred_name("   ", {}) == ""


# ---- wire payload ------------------------------------------------------
def test_profile_payload_shape():
    payload = P.profile_payload(
        _User(name="Kiran Jana", preferences={"display_name": "KJ",
                                              "avatar": PNG}))
    assert set(payload) == {"id", "email", "full_name", "display_name",
                            "avatar", "limits"}
    assert payload["full_name"] == "Kiran Jana"
    assert payload["display_name"] == "KJ"
    assert payload["avatar"] == PNG
    assert payload["limits"]["display_name"] == P.MAX_DISPLAY_NAME_CHARS


def test_profile_payload_of_a_bare_user():
    payload = P.profile_payload(_User())
    assert payload["full_name"] == ""
    assert payload["display_name"] == ""
    assert payload["avatar"] == ""


def test_profile_payload_never_leaks_other_preferences():
    payload = P.profile_payload(
        _User(preferences={"custom_instructions": "secret-ish", "api_key": "x"}))
    assert "custom_instructions" not in payload
    assert "api_key" not in payload


# ---- the PATCH semantics ----------------------------------------------
def _patch(user, **fields):
    """Replay the route's field-by-field application (no DB needed)."""
    from app.api.routes_auth import ProfileUpdate
    body = ProfileUpdate(**fields)
    rejected: list[str] = []
    if body.full_name is not None:
        user.name = P.clean_full_name(body.full_name) or None
    if body.display_name is not None:
        user.preferences = P.set_display_name(user.preferences, body.display_name)
    if body.avatar is not None:
        raw = body.avatar.strip()
        if raw and not P.clean_avatar(raw):
            rejected.append("avatar")
        else:
            user.preferences = P.set_avatar(user.preferences, raw)
    return rejected


def test_an_omitted_field_is_left_alone():
    user = _User(name="Kiran Jana", preferences={"display_name": "KJ",
                                                 "avatar": PNG})
    _patch(user, full_name="Kiran J")
    assert user.name == "Kiran J"
    # Untouched, because they weren't in the body at all.
    assert P.load_display_name(user.preferences) == "KJ"
    assert P.load_avatar(user.preferences) == PNG


def test_an_explicit_empty_string_clears():
    user = _User(name="Kiran", preferences={"display_name": "KJ", "avatar": PNG})
    _patch(user, display_name="", avatar="")
    assert P.load_display_name(user.preferences) == ""
    assert P.load_avatar(user.preferences) == ""


def test_a_rejected_avatar_does_not_block_the_rest_of_the_save():
    user = _User(name="old")
    rejected = _patch(user, full_name="Kiran Jana",
                      avatar="data:image/svg+xml;base64,AAAA")
    assert rejected == ["avatar"]
    assert user.name == "Kiran Jana"      # the name still saved
    assert P.load_avatar(user.preferences) == ""


def test_clearing_the_full_name_stores_null_not_empty_string():
    # `User.name` is nullable; an empty string would be a second "no name" value.
    user = _User(name="Kiran")
    _patch(user, full_name="   ")
    assert user.name is None


# ---- routes + prompt wiring -------------------------------------------
@pytest.mark.parametrize("path", ["/api/auth/profile"])
def test_profile_routes_registered(path):
    from app.api import routes_auth
    paths = {r.path for r in routes_auth.router.routes}
    assert path in paths
    methods = {m for r in routes_auth.router.routes if r.path == path
               for m in r.methods}
    assert {"GET", "PATCH"} <= methods


def test_the_preferred_name_is_framed_for_the_prompt():
    # The field would be pointless if it were only ever displayed back to the
    # user — the assistant has to be told. Pure helper, like frame_instructions.
    block = P.frame_preferred_name("Kiran")
    assert "Kiran" in block
    # …and told to be restrained about it, or it opens every reply with the name.
    assert "every reply" in block


def test_no_preferred_name_frames_nothing():
    for blank in (None, "", "   ", 42):
        assert P.frame_preferred_name(blank) == ""


def test_the_persona_prompt_uses_the_framer():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "agents"
           / "persona.py").read_text(encoding="utf-8", errors="replace")
    assert "frame_preferred_name" in src
    assert 'extras.get("preferred_name")' in src


def test_the_stream_route_populates_preferred_name():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "api"
           / "routes_agents.py").read_text(encoding="utf-8", errors="replace")
    assert "from app.personalization.profile import preferred_name" in src
    assert 'extras_base["preferred_name"]' in src


def _run(coro):
    return asyncio.run(coro)


def test_profile_endpoints_require_auth():
    from fastapi import HTTPException
    from app.api import routes_auth

    class _NoSession:
        async def get(self, *_a, **_k):
            return None

    with pytest.raises(HTTPException) as err:
        _run(routes_auth.get_profile(session=_NoSession()))
    assert err.value.status_code == 401
