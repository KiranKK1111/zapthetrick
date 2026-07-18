"""Interview state machine + hypothesis buffer / turn-taking
(live-conversational-intelligence R3, R4; task 2.2).

Pins Properties 4-5: defined transitions, concurrency preserved (two close
questions stay ANSWERING with independent answers), and settle/merge
turn-taking. Pure/deterministic — time is injected.
"""
from __future__ import annotations

from app.live import events
from app.live.hypothesis import HypothesisBuffer
from app.live.state_machine import (
    ANSWERING,
    CONTEXT_BUILDING,
    FOLLOWUP_WAITING,
    IDLE,
    QUESTION_CONFIRMED,
    TOPIC_SWITCHING,
    InterviewStateMachine,
    get_state_machine,
)


def _q(kind=events.QUESTION, questions=None):
    return events.UtteranceEvent(kind=kind, questions=questions or ["q?"])


# ---- state machine -----------------------------------------------------
def test_starts_idle():
    assert InterviewStateMachine().state == IDLE


def test_question_confirms():
    sm = InterviewStateMachine()
    assert sm.advance(_q()) == QUESTION_CONFIRMED


def test_explanation_builds_context():
    sm = InterviewStateMachine()
    assert sm.advance(_q(kind=events.EXPLANATION, questions=[])) == CONTEXT_BUILDING


def test_topic_change_switches():
    sm = InterviewStateMachine()
    assert sm.advance(_q(kind=events.TOPIC_CHANGE, questions=[])) == TOPIC_SWITCHING


def test_concurrency_preserved():
    """Two questions answered concurrently stay ANSWERING until BOTH finish —
    the machine never blocks the second answer."""
    sm = InterviewStateMachine()
    sm.advance(_q())
    sm.on_answer_start()              # qid A
    assert sm.state == ANSWERING
    # A second question arrives while A is still streaming.
    sm.advance(_q())
    assert sm.state == ANSWERING      # not reset / blocked
    sm.on_answer_start()              # qid B
    sm.on_answer_done()               # A done, B still streaming
    assert sm.state == ANSWERING
    sm.on_answer_done()               # B done
    assert sm.state == FOLLOWUP_WAITING


def test_snapshot_shape():
    sm = InterviewStateMachine()
    sm.advance(_q())
    snap = sm.snapshot()
    assert snap["state"] == QUESTION_CONFIRMED
    assert snap["active_answers"] == 0
    assert "last_event" in snap


def test_registry_is_per_session():
    a = get_state_machine("s1")
    b = get_state_machine("s1")
    c = get_state_machine("s2")
    assert a is b
    assert a is not c


# ---- hypothesis buffer / turn-taking -----------------------------------
def test_not_due_before_settle():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("How would you scale Kafka", now=0.0)
    assert not buf.settle_due(now=0.3)     # 300ms < 600ms
    assert buf.settle_due(now=0.7)         # 700ms >= 600ms


def test_continuation_merges_into_same_hypothesis():
    buf = HypothesisBuffer(settle_ms=600)
    g1 = buf.add("How would you scale Kafka", now=0.0)
    # Speaker continues within the window -> merge, generation bumps.
    g2 = buf.add("for a system with 100 million users?", now=0.4)
    assert g2 > g1
    assert not buf.settle_due(now=0.5)     # window restarted from 0.4
    text, _ = buf.take()
    assert "100 million" in text
    assert text.startswith("How would you scale Kafka")


def test_dynamic_endpointing_complete_settles_fast():
    from app.live.hypothesis import completeness
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("How would you handle event ordering in Kafka?", now=0.0)
    assert completeness(buf.merged()) == "complete"
    # Complete question → shorter wait (0.6x, floored at 250ms).
    assert buf.required_settle_ms() == 360
    assert buf.settle_due(now=0.4)         # 400ms >= 360ms → answer promptly


def test_dynamic_endpointing_incomplete_waits_longer():
    from app.live.hypothesis import completeness
    buf = HypothesisBuffer(settle_ms=600)
    # Interviewer paused mid-thought ("...and how would you") — do NOT cut.
    buf.add("We use Kafka heavily and", now=0.0)
    assert completeness(buf.merged()) == "incomplete"
    assert buf.required_settle_ms() == 2100         # 600 * 3.5
    assert not buf.settle_due(now=1.0)     # 1000ms < 2100ms → keep waiting
    assert not buf.settle_due(now=2.0)
    # Speaker finishes the question within the extended window → merges.
    buf.add("how would you scale it?", now=1.1)
    assert completeness(buf.merged()) == "complete"


def test_dynamic_endpointing_question_stem_gap_not_split():
    """The reported bug: 'What is <pause> Kafka?' must NOT split. The bare
    stem 'What is' reads incomplete → long settle → the object merges in."""
    from app.live.hypothesis import completeness
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("What is", now=0.0)
    assert completeness("What is") == "incomplete"
    assert buf.required_settle_ms() == 2100
    # 1.5s thinking pause — still within the window, not cut.
    assert not buf.settle_due(now=1.5)
    buf.add("Kafka?", now=1.6)
    text, _ = buf.take()
    assert text == "What is Kafka?"


def test_completeness_object_present_is_not_incomplete():
    """A complete short question ('What is microservices') must NOT be treated
    as incomplete (that would add a needless 2s delay to every such answer)."""
    from app.live.hypothesis import completeness
    assert completeness("What is microservices") == "neutral"
    assert completeness("what motivates you") == "neutral"
    assert completeness("tell me about your project") == "neutral"


def test_take_clears_buffer():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("What is Kafka?", now=0.0)
    assert buf.pending()
    buf.take()
    assert not buf.pending()


def test_zero_settle_is_always_due():
    buf = HypothesisBuffer(settle_ms=0)
    buf.add("What is Kafka?", now=0.0)
    assert buf.settle_due(now=0.0)


def test_audio_flag_tracked():
    buf = HypothesisBuffer(settle_ms=300)
    buf.add("part one", now=0.0, has_audio=True)
    buf.add("part two", now=0.1, has_audio=False)
    _, had_audio = buf.take()
    assert had_audio is True
