"""Phrase-level transcript repair (live-conversational-intelligence R11).

Pins the sliding n-gram phrase pass: melted domain terms ("my service" →
"microservices", "cube net is ingress" → "kubernetes ingress") are repaired,
while legitimate common-word phrases and exact vocabulary matches are never
touched. Also pins idempotence and casing/punctuation preservation.
"""
from __future__ import annotations

from app.live import repair


# ---- phrase repairs that MUST happen ------------------------------------
def test_phrase_my_service_becomes_microservices():
    # "microservices" is in the built-in vocab; punctuation is preserved.
    assert repair.repair("what is my service?") == "what is microservices?"


def test_phrase_cube_net_is_ingress():
    out = repair.repair("explain cube net is ingress")
    assert "kubernetes ingress" in out.lower()


def test_phrase_preserves_leading_capitalization():
    out = repair.repair("My service is great")
    assert out.startswith("Microservices")


# ---- cases that must NOT be corrected -----------------------------------
def test_no_correction_service_level_agreement():
    # The exact vocab phrase protects its span from fuzzy phrase rewrites.
    text = "what is my service level agreement"
    assert repair.repair(text, vocab=["service level agreement"]) == text


def test_no_correction_plain_common_sentence():
    text = "what do you think about the design"
    assert repair.repair(text) == text


def test_no_correction_correct_phrase_untouched():
    text = "explain kubernetes ingress"
    assert repair.repair(text) == text


def test_no_correction_grammatical_inflection():
    # "unit tests" is grammar, not a mishearing of "unit testing".
    text = "we should write unit tests"
    assert repair.repair(text) == text


# ---- idempotence ----------------------------------------------------------
def test_phrase_repair_is_idempotent():
    once = repair.repair("what is my service?")
    assert repair.repair(once) == once

    once = repair.repair("explain cube net is ingress")
    assert repair.repair(once) == once
