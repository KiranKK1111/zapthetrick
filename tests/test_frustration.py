"""Frustration detection (personalization-and-governance R3, task 4.2).

Pins Property 3: rise on correction/negative/rephrase, decay to baseline,
fewer-questions bias while elevated, and a safety exemption (the bias is never a
safety override).
"""
from __future__ import annotations

from app.personalization.frustration import (
    FrustrationState, update_frustration, bias,
)


def test_rises_on_correction():
    s = update_frustration(FrustrationState(), "no, that's wrong, I meant X")
    assert s.value > 0.0


def test_rises_on_negative_feedback_flag():
    s = update_frustration(FrustrationState(), "ok", negative_feedback=True)
    assert s.value >= 0.4


def test_rises_on_rephrase():
    s = update_frustration(FrustrationState(),
                           "how do I configure the database connection",
                           prev_turn="how to configure the database connection")
    assert s.value > 0.0


def test_decays_when_normal():
    s = FrustrationState(value=0.8)
    s2 = update_frustration(s, "thanks, that makes sense")
    assert s2.value < 0.8


def test_elevated_biases_fewer_questions():
    s = FrustrationState(value=0.7)
    b = bias(s)
    assert b.get("prefer_concise") and b.get("fewer_clarifications")


def test_not_elevated_no_bias():
    assert bias(FrustrationState(value=0.2)) == {}


def test_bias_is_additive_not_a_safety_override():
    # The bias dict carries no field that could suppress a safety confirmation;
    # it only nudges verbosity/clarifications (R3.4).
    b = bias(FrustrationState(value=0.9))
    assert "suppress_safety" not in b and "skip_confirmation" not in b
