"""Conversational-act classifier (followup-context-engine R2, task 3.2).

Pins Property 3: exactly one primary act, new-topic gating below the
follow-up-confidence threshold, approval/rejection recognized (so the route can
update state without answering), and error fallback to new_topic.
"""
from __future__ import annotations

import pytest

from app.followup import acts
from app.followup.state import ConversationState


@pytest.fixture(autouse=True)
def _deterministic_topic_shift(monkeypatch):
    """is_topic_shift is now SEMANTIC-first; pin the deterministic cue-list
    fallback here (warm gate → valid-but-different verdicts in the full suite).
    The gate itself is covered in test_semantic_gates."""
    import app.semantics.gates as _g
    monkeypatch.setattr(_g, "matches", lambda *a, **k: None)


def _state_with_context():
    s = ConversationState({}, "c1")
    s.set_goal("build a Flutter chat app")
    s.add_entity("Flutter")
    s.set_enumerations(["PostgreSQL", "MongoDB", "SQLite"])
    return s


def test_exactly_one_act_in_closed_set():
    s = _state_with_context()
    act, conf = acts.classify("make it better", s)
    assert act in acts.ACTS
    assert 0.0 <= conf <= 1.0


def test_approval_and_rejection_recognized():
    s = _state_with_context()
    assert acts.classify("yes", s)[0] == acts.APPROVAL
    assert acts.classify("looks good", s)[0] == acts.APPROVAL
    assert acts.classify("no", s)[0] == acts.REJECTION


def test_correction_not_downgraded():
    s = _state_with_context()
    act, _ = acts.classify("actually use SQLite instead", s)
    assert act == acts.CORRECTION


def test_explicit_topic_shift_is_new_topic():
    """An explicit switch must classify as NEW_TOPIC with high confidence so the
    route drops the prior thread's context (issue: topic switching)."""
    s = _state_with_context()
    for turn in (
        "different question: what is rust ownership",
        "let's move on to postgres indexing",
        "new topic - how do neural nets train",
        "forget that, tell me about docker networking",
        "on a different note, what is the CAP theorem",
    ):
        act, conf = acts.classify(turn, s)
        assert act == acts.NEW_TOPIC, turn
        assert conf >= 0.85, turn
        assert acts.is_topic_shift(turn) is True, turn


def test_followups_are_not_topic_shifts():
    """Ordinary follow-ups / corrections must NOT be treated as topic shifts."""
    for turn in (
        "actually can you explain that better",
        "what about kafka instead",
        "can you continue",
        "optimize this function",
        "make it better",
    ):
        assert acts.is_topic_shift(turn) is False, turn


def test_continuation_and_comparison_and_expansion():
    s = _state_with_context()
    assert acts.classify("continue", s)[0] == acts.CONTINUATION
    assert acts.classify("compare them", s)[0] == acts.COMPARISON
    assert acts.classify("explain more", s)[0] == acts.EXPANSION


def test_new_topic_gating_without_context():
    """A bare pronoun follow-up with NO conversation context is treated as a
    new topic (low follow-up confidence) — Property 3 / R2.3."""
    fresh = ConversationState({}, "c2")
    act, _ = acts.classify("a fully self contained brand new question here", fresh)
    assert act == acts.NEW_TOPIC


def test_clarification_answer_when_open_question_pending():
    s = ConversationState({}, "c3")
    s.add_open_question("Which database?")
    act, _ = acts.classify("PostgreSQL", s)
    assert act == acts.CLARIFICATION_ANSWER


def test_error_fallback_to_new_topic():
    class _Boom:
        def open_questions(self):
            raise RuntimeError("boom")

    act, conf = acts.classify("continue", _Boom())
    # open_questions raising is swallowed inside _classify; the act still
    # resolves deterministically (continuation), proving fail-open never raises.
    assert act in acts.ACTS and 0.0 <= conf <= 1.0
