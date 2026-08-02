"""Solo mode: distinguish a candidate DELIVERING an answer from one ASKING.

Standard mode separates speakers by role. Solo deliberately cannot — one voice,
nothing to diarize — so its only protection was echo matching, which catches the
tester reading our shown answer back. A candidate answering in their OWN words
matched nothing and was sent to the LLM as a new question, so the app
interrupted the person it exists to help.

The asymmetry drives every decision here: wrongly answering one sentence of a
delivery is annoying; wrongly suppressing a real question leaves someone silent
in an interview. So a strong grammatical question signal ALWAYS wins, and
anything unclassified is admitted.
"""
from __future__ import annotations

import pytest

from app.live import delivery_state as D


def state(topic: str = "kafka partitions", shown: bool = True) -> D.DeliveryState:
    st = D.DeliveryState(window_s=25.0)
    if shown:
        st.answer_shown(topic, now=1000.0)
    return st


NOW = 1005.0        # 5 s into the window
LATER = 1100.0      # well outside it


# ── The reported gap: answering in your own words ───────────────────────────

@pytest.mark.parametrize("said", [
    "I used partitions to parallelise the consumer group in my last role",
    "So what I did was shard the topic by customer id",
    "We handled it by increasing the partition count",
    "In my previous project we hit exactly this problem",
    "My approach was to key the messages by order id",
])
def test_first_person_delivery_is_suppressed(said):
    suppress, why = D.should_suppress(state(), said, now=NOW)
    assert suppress, f"would have answered the candidate's own answer: {why}"


def test_thinking_aloud_is_suppressed():
    suppress, _ = D.should_suppress(state(), "hmm let me think about that", now=NOW)
    assert suppress


def test_speech_continuing_the_answered_topic_is_suppressed():
    suppress, why = D.should_suppress(
        state("kafka partitions"), "partitions are the unit of parallelism",
        now=NOW)
    assert suppress and "topic" in why


# ── The inverse, which matters more ─────────────────────────────────────────

@pytest.mark.parametrize("said", [
    "What is a consumer group?",
    "How would you handle a rebalance",          # no '?', wh-lead
    "Does that scale horizontally",              # inversion
    "And what about exactly-once delivery",      # follow-up fragment
    "Write a function that partitions the keys", # imperative prompt
    "Say the broker dies, where do you start",   # clause-leading interrogative
])
def test_a_real_question_is_ALWAYS_admitted_even_mid_delivery(said):
    """A strong grammatical signal wins over every delivery heuristic. Leaving a
    real question unanswered is the failure that actually costs an interview."""
    suppress, why = D.should_suppress(state(), said, now=NOW)
    assert not suppress, f"suppressed a real question ({why}): {said!r}"


def test_a_first_person_QUESTION_is_still_admitted():
    """"I" does not make it an answer — the grammar does."""
    suppress, _ = D.should_suppress(
        state(), "I'm curious, how does rebalancing work?", now=NOW)
    assert not suppress


def test_an_unclassified_utterance_is_admitted():
    """No question signal, but nothing marks it as delivery either. Answering an
    unclear utterance is safer than swallowing it."""
    suppress, why = D.should_suppress(state("kafka"), "right, okay then", now=NOW)
    assert not suppress and "unclassified" in why


# ── The window ──────────────────────────────────────────────────────────────

def test_nothing_is_suppressed_outside_the_window():
    suppress, why = D.should_suppress(
        state(), "I used partitions in my last role", now=LATER)
    assert not suppress and "window" in why


def test_nothing_is_suppressed_before_any_answer_was_shown():
    """At the very start of a session everything is a question."""
    suppress, _ = D.should_suppress(state(shown=False), "I think so", now=NOW)
    assert not suppress


def test_a_new_answer_reopens_the_window():
    st = state()
    assert not D.should_suppress(st, "I used partitions", now=LATER)[0]
    st.answer_shown("redis", now=LATER)
    assert D.should_suppress(st, "I used redis for caching", now=LATER + 2)[0]


# ── Mechanics ───────────────────────────────────────────────────────────────

def test_counters_make_a_suppressed_turn_explainable():
    st = state()
    D.should_suppress(st, "I did X in my last role", now=NOW)
    D.should_suppress(st, "What is a partition?", now=NOW)
    assert st.suppressed == 1 and st.admitted == 1


def test_empty_speech_is_a_no_op():
    assert D.should_suppress(state(), "   ", now=NOW) == (False, "")


def test_strong_signal_detection_fails_open(monkeypatch):
    """If the classifier is unavailable, treat speech as a QUESTION — the safe
    direction."""
    import app.question_detection.classifier as C
    monkeypatch.delattr(C, "_clauses", raising=False)
    assert D.has_strong_question_signal("I used partitions") is True


def test_topic_continuity_ignores_short_words():
    assert D.topic_continues("the of and", "the of and") is False
    assert D.topic_continues("partitions rebalance", "kafka partitions") is True


def test_sessions_are_isolated():
    a, b = D.state_for("s1"), D.state_for("s2")
    a.answer_shown("kafka")
    assert b.shown_at == 0.0
    D.forget("s1")
    D.forget("s2")


# ── Wiring ──────────────────────────────────────────────────────────────────

def test_the_solo_path_consults_the_delivery_state():
    import inspect

    from app.api import routes_ws
    src = inspect.getsource(routes_ws)
    assert "delivery_state" in src, "the solo path never consults it"
    assert "candidate_delivery" in src, "no skip reason is surfaced"
    assert "answer_shown" in src, "the window is never opened"
