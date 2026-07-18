"""Structured event typer (live-conversational-intelligence R1, R2; task 1.2).

Pins Properties 1-3: event typing, question-boundary split, multi-question
split, legacy-shape mapping, and fail-open to today's single-question decision.
Uses an injected fake predictor so no LLM is called.
"""
from __future__ import annotations

import asyncio

from app.live import events
from app.question_detection.agent import Prediction


def _run(coro):
    return asyncio.run(coro)


def _fake(pred: Prediction):
    async def _predict(text, recent):  # noqa: ANN001
        return pred
    return _predict


# ---- deterministic split helpers (no LLM) ------------------------------
def test_split_questions_multi_explicit():
    qs = events.split_questions("What is Kafka? Why use it? How do you scale it?")
    assert len(qs) == 3
    assert all(q.endswith("?") for q in qs)


def test_split_questions_conjoined():
    qs = events.split_questions("What is Kafka, why is it used, and how would you scale it?")
    assert len(qs) >= 2
    assert any("scale" in q.lower() for q in qs)


def test_split_questions_single():
    assert events.split_questions("How do partitions work?") == ["How do partitions work?"]


def test_split_questions_empty():
    assert events.split_questions("   ") == []


def test_split_boundary_separates_context():
    ctx, q = events.split_boundary(
        "We use Kafka. Ordering matters. How do you handle duplicates?",
        "How do you handle duplicates?",
    )
    assert len(ctx) == 2
    assert "Kafka" in ctx[0]
    assert q.lower().startswith("how do you handle")


def test_split_boundary_single_sentence():
    ctx, q = events.split_boundary("What is polymorphism?", "What is polymorphism?")
    assert ctx == []
    assert q == "What is polymorphism?"


# ---- type_utterance (reuses the one agent.predict call) ----------------
def test_question_event_with_boundary_and_questions():
    pred = Prediction(True, "How do you handle duplicate events?", "technical_concept",
                      "kafka", "standard")
    ev = _run(events.type_utterance(
        "We use Kafka. Ordering matters. How do you handle duplicate events?",
        [], predictor=_fake(pred)))
    assert ev.kind == events.QUESTION
    assert ev.is_answerable
    assert ev.questions and "duplicate" in ev.questions[0].lower()
    assert any("Kafka" in c for c in ev.context)
    assert ev.topic == "kafka"


def test_multi_question_utterance_splits():
    pred = Prediction(True, "What is Kafka, why is it used, and how would you scale it?",
                      "technical_concept", "kafka", "standard")
    ev = _run(events.type_utterance(
        "What is Kafka, why is it used, and how would you scale it?", [],
        predictor=_fake(pred)))
    assert ev.kind == events.QUESTION
    assert len(ev.questions) >= 2


def test_non_question_is_explanation_not_answerable():
    pred = Prediction(False, "", "technical_concept", "kafka", "standard")
    ev = _run(events.type_utterance("In our company we use Kafka extensively.", [],
                                    predictor=_fake(pred)))
    assert ev.kind == events.EXPLANATION
    assert not ev.is_answerable
    assert ev.questions == []


def test_smalltalk_maps_to_small_talk():
    pred = Prediction(False, "", "smalltalk", "", "standard")
    ev = _run(events.type_utterance("Yeah, sounds good.", [], predictor=_fake(pred)))
    assert ev.kind in (events.SMALL_TALK, events.ACKNOWLEDGEMENT)
    assert not ev.is_answerable


def test_greeting_refined():
    pred = Prediction(False, "", "smalltalk", "", "standard")
    ev = _run(events.type_utterance("Good morning, nice to meet you.", [],
                                    predictor=_fake(pred)))
    assert ev.kind == events.GREETING


def test_fail_open_when_predictor_raises():
    async def _boom(text, recent):  # noqa: ANN001
        raise RuntimeError("model down")
    ev = _run(events.type_utterance("Explain the CAP theorem.", [], predictor=_boom))
    assert ev.kind == events.QUESTION
    assert ev.questions == ["Explain the CAP theorem."]
    assert ev.source == "fallback"


def test_empty_utterance_is_noop_event():
    ev = _run(events.type_utterance("   ", [], predictor=_fake(Prediction(True, "x", "coding", ""))))
    assert ev.kind == events.SMALL_TALK
    assert not ev.is_answerable
