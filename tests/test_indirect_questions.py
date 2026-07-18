"""Indirect / hypothetical / tonal question detection (live path).

Interview questions are frequently not phrased as questions:
* indirect imperatives — "Walk me through your project."
* hypothetical scenarios — "Suppose one service goes down."
* tonal questions — statement words, rising terminal pitch.

Pins: the hypothetical detector (pronoun-guarded), the decision engine's
promotion of not-answerable events, the heuristic fallback agreeing with
the promotion layers, and the cleaned-up intent header labels.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.live import decision
from app.live.implicit import detect_hypothetical, detect_implicit


# ── Hypothetical detector ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Suppose one of your services goes down.",
    "Let's say the database crashes at peak traffic.",
    "Imagine your API is suddenly slow.",
    "What if we double the traffic overnight?",
    "Now suppose the cache is cold.",
    "Assume you have a million concurrent users.",
    "Hypothetically, the deploy fails halfway.",
])
def test_hypothetical_probes_detected(text):
    sig = detect_hypothetical(text)
    assert sig.is_implicit_question, text
    assert sig.confidence >= 0.6


@pytest.mark.parametrize("text", [
    "I suppose that's fine.",           # hedge, not a probe
    "We assume standard latency here.",  # speaker's own assumption
    "I imagine that was difficult.",     # empathy, not a scenario
    "The weather is nice today.",
    "",
])
def test_hedges_and_statements_are_not_hypothetical(text):
    assert not detect_hypothetical(text).is_implicit_question, text


# ── Decision-engine promotion of not-answerable events ───────────────────────

def _event(answerable: bool = False):
    return SimpleNamespace(is_answerable=answerable, kind="STATEMENT",
                           questions=[], context=[], topic="")


def test_indirect_probe_promoted_instead_of_skipped():
    d = decision.decide_event(
        _event(), is_audio=True,
        utterance="Walk me through your project architecture")
    assert d.action == decision.ANSWER
    assert d.signals.get("promoted") == "implicit"


def test_hypothetical_probe_promoted_instead_of_skipped():
    d = decision.decide_event(
        _event(), is_audio=True,
        utterance="Suppose one of your services goes down.")
    assert d.action == decision.ANSWER
    assert d.signals.get("promoted") == "hypothetical"
    assert d.signals.get("promoted_qtype") == "hypothetical"


def test_plain_statement_still_skipped():
    d = decision.decide_event(
        _event(), is_audio=True,
        utterance="We were founded in 2015 and have six offices.")
    assert d.action == decision.SKIP


def test_typed_input_unaffected_by_promotion_path():
    # Non-audio (typed) input never hits the answerability rule.
    d = decision.decide_event(
        _event(), is_audio=False, utterance="anything at all")
    assert d.action == decision.ANSWER


def test_decide_utterance_emits_hypothetical_signal():
    d = decision.decide_utterance(
        "Let's say traffic doubles overnight.", is_audio=True)
    assert "hypothetical" in d.signals
    assert any(f.get("qtype") == "hypothetical" for f in d.frames)


# ── Heuristic fallback agrees ────────────────────────────────────────────────

def test_heuristic_fallback_catches_indirect_and_hypothetical():
    from app.question_detection.classifier import heuristic_classify
    assert heuristic_classify(
        "Suppose one of your services goes down.").is_question
    assert heuristic_classify(
        "I'd like to hear how you handled that outage.").is_question
    assert not heuristic_classify(
        "The office is on the third floor.").is_question


# ── Intent header labels ─────────────────────────────────────────────────────

def test_intent_label_drops_empty_topics():
    from app.api.routes_ws import _intent_label
    # The reported "Question — unknown" header must never render.
    assert _intent_label("unknown", topic="unknown") == "Question"
    assert _intent_label(None, topic="") == "Question"
    assert _intent_label("technical_concept", topic="unknown") == (
        "Technical concept")
    assert _intent_label("technical_concept", topic="kafka") == (
        "Technical concept — kafka")
    assert _intent_label("hypothetical") == "Scenario question"


def test_implicit_detector_unchanged():
    # Regression guard: the existing implicit layer still works.
    assert detect_implicit("Walk me through your approach").is_implicit_question
    assert not detect_implicit("What is a hash map?").is_implicit_question
