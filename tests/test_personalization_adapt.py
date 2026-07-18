"""Response adaptation bridge (personalization-and-governance R2, task 3.2).

Pins Property 2: the user model biases concise/detailed via the existing
answer-depth signal, but never overrides an explicit per-turn depth choice.
Reuses `AnswerDepthPolicy` conceptually — here the bridge maps verbosity → the
depth-mechanic preference and respects explicit choice.
"""
from __future__ import annotations

from app.personalization.user_model import UserModel
from app.personalization.adapt import preferred_depth


def test_concise_pref_biases_concise():
    m = UserModel(verbosity_pref="concise")
    assert preferred_depth(m, explicit=None) == "tldr"


def test_detailed_pref_biases_detailed():
    m = UserModel(verbosity_pref="detailed")
    assert preferred_depth(m, explicit=None) in ("deeper", "exhaustive")


def test_explicit_choice_always_wins():
    m = UserModel(verbosity_pref="concise")
    # An explicit per-turn depth choice is never overridden (R2.3).
    assert preferred_depth(m, explicit="exhaustive") == "exhaustive"


def test_neutral_model_no_bias():
    assert preferred_depth(UserModel(), explicit=None) is None


def test_expert_biases_concise_when_no_verbosity():
    m = UserModel(expertise="expert")
    assert preferred_depth(m, explicit=None) == "tldr"
