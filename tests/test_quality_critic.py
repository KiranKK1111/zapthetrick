"""Response quality critic (evaluation-and-reliability R7, task 10.3).

Pins Property 8: deterministic coverage + contradiction detection, non-blocking
(sync, no model), and a clean skip when data is missing.
"""
from __future__ import annotations

import inspect

from app.quality.critic import review, CriticReport


def test_full_coverage_no_gaps():
    rep = review(
        "Here is a Python function that reverses a string using slicing.",
        asked_items=["reverse a string in python"],
    )
    assert isinstance(rep, CriticReport)
    assert rep.covered and not rep.gaps and not rep.skipped


def test_detects_uncovered_asked_item():
    rep = review(
        "Here is how to reverse a string.",
        asked_items=["reverse a string", "also add unit tests with pytest"],
    )
    assert not rep.covered
    assert any("unit tests" in g for g in rep.gaps)


def test_detects_decision_contradiction():
    rep = review(
        "I recommend storing the data in MySQL with a normalized schema.",
        asked_items=["design the data layer"],
        decisions={"database": "postgres"},
    )
    assert rep.contradictions
    assert any("mysql" in c.lower() for c in rep.contradictions)


def test_no_contradiction_when_decided_value_present():
    rep = review(
        "We'll use PostgreSQL for storage as decided.",
        asked_items=["design the data layer"],
        decisions={"database": "postgres"},
    )
    assert rep.contradictions == []


def test_skips_cleanly_when_no_inputs():
    assert review("", asked_items=[], decisions={}).skipped is True
    assert review("some answer", asked_items=None, decisions=None).skipped is True


def test_review_is_synchronous_non_blocking():
    # Non-blocking: the critic is a plain sync function, never an LLM call.
    assert not inspect.iscoroutinefunction(review)


def test_review_failopen_never_raises():
    # Malformed decisions value types must not raise (Property 8/9).
    rep = review("answer text", asked_items=["x"], decisions={"k": None})
    assert isinstance(rep, CriticReport)
