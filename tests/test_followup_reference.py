"""Reference resolution (followup-context-engine R3/R4, task 4.3).

Pins Properties 4 & 5: pronoun + selection + entity resolution, ordinal
correctness against the most recent enumerations, confidence-gated deferral to
the clarifier, and no-antecedent fallback.
"""
from __future__ import annotations

from app.followup.reference import resolve
from app.followup.state import ConversationState


def _state():
    s = ConversationState({}, "c1")
    s.add_entity("Flutter")
    s.add_entity("PostgreSQL")
    s.set_enumerations(["Redis", "Memcached", "Dragonfly"])
    return s


def test_selection_reference_ordinal_correctness():
    s = _state()
    r = resolve("use the second one", s)
    assert r.resolved
    assert r.antecedents == ["Memcached"]      # second of the enumeration
    r2 = resolve("go with the last one", s)
    assert r2.antecedents == ["Dragonfly"]


def test_option_letter_reference():
    s = _state()
    r = resolve("option C please", s)
    assert r.antecedents == ["Dragonfly"]      # C → index 2


def test_entity_reference_resolves_from_registry():
    s = _state()
    r = resolve("make the PostgreSQL schema stricter", s)
    assert "PostgreSQL" in r.antecedents
    assert r.confidence >= 0.6


def test_pronoun_resolves_to_most_salient_entity():
    s = _state()                                # most-recent entity = PostgreSQL
    r = resolve("optimize it", s)
    assert r.antecedents == ["PostgreSQL"]


def test_low_confidence_defers_to_clarifier():
    """An ordinal with NO enumerations to resolve against → low confidence →
    defer to the clarifier rather than guess (R3.3 / Property 4)."""
    s = ConversationState({}, "c2")             # no enumerations, no entities
    r = resolve("do the second one", s)
    assert r.needs_clarification is True
    assert r.confidence < 0.6


def test_no_reference_leaves_turn_untouched():
    s = _state()
    r = resolve("build a brand new REST API", s)
    assert not r.resolved
    assert r.refs == [] and r.confidence == 0.0
    assert r.needs_clarification is False
