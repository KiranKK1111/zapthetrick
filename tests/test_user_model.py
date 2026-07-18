"""User model (personalization-and-governance R1, task 1.2).

Pins Property 1: inference, neutral default, depth-preference reuse, persistence
(no new schema), and data-clear.
"""
from __future__ import annotations

from app.personalization.user_model import (
    UserModel, infer, load_user_model, save_user_model, clear_user_model, UNKNOWN,
)


def test_neutral_default_on_no_signal():
    m = infer({})
    assert m.is_neutral
    assert m.expertise == UNKNOWN and m.verbosity_pref == UNKNOWN


def test_verbosity_reuses_depth_pref():
    assert infer({"depth_pref": "tldr"}).verbosity_pref == "concise"
    assert infer({"depth_pref": "exhaustive"}).verbosity_pref == "detailed"
    assert infer({"depth_pref": "standard"}).verbosity_pref == "balanced"


def test_expertise_from_cues():
    senior = infer({"recent_user_texts": ["optimize the time complexity and "
                                          "avoid the race condition"]})
    assert senior.expertise == "senior"
    beginner = infer({"recent_user_texts": ["what is a variable, explain like "
                                            "i'm new"]})
    assert beginner.expertise == "beginner"


def test_explicit_expertise_hint_wins():
    assert infer({"expertise_hint": "expert"}).expertise == "expert"


def test_comm_style_from_bullet_ratio():
    assert infer({"bullet_ratio": 0.8}).comm_style == "bullet"
    assert infer({"bullet_ratio": 0.1}).comm_style == "prose"


def test_persistence_round_trip_no_schema():
    prefs: dict = {}
    m = UserModel(expertise="senior", verbosity_pref="concise",
                  comm_style="technical", frustration=0.3)
    save_user_model(prefs, m)
    assert "user_model" in prefs
    loaded = load_user_model(prefs)
    assert loaded.expertise == "senior" and loaded.verbosity_pref == "concise"


def test_data_clear_removes_model():
    prefs = {"user_model": {"expertise": "senior"}}
    clear_user_model(prefs)
    assert "user_model" not in prefs
    assert load_user_model(prefs).is_neutral


def test_infer_failopen():
    assert isinstance(infer(None), UserModel)
