"""Live question-detection quality gate, measured on the 4213-row corpus.

Why this exists as a TEST and not just a script: the seed corpus was 32 rows,
where a single utterance moved F1 by ~3 points. At that size no measured
difference between two builds could be trusted, so detection quality silently
drifted. These thresholds turn "it feels about as good" into a build failure.

The thresholds are set just under the measured values, so an honest improvement
never fails the build but a regression does. Raise them when you beat them.

Recall is the metric that matters most here. A missed question is not a
degraded answer — it is **silence** in a live interview, which is the worst
outcome the product has. Precision failures merely produce an answer nobody
needed.
"""
from __future__ import annotations

import pathlib

import pytest

from app.eval.live_bench import run_corpus

_CORPUS = (pathlib.Path(__file__).resolve().parents[1]
           / "app" / "eval" / "data" / "live_corpus_large.jsonl")


def _embedder_ready() -> bool:
    """The semantic gates need the embedder. Without it the deterministic floor
    is what runs — which is a real deployment state (a cold pod), so the
    thresholds below are split into two tiers rather than skipping outright."""
    try:
        from app.rag import embedder
        embedder.embed(["warm"])
        return bool(embedder.is_ready())
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def report():
    # Warm BEFORE measuring. Without this the corpus runs against a cold
    # embedder (deterministic floor only) while the threshold picker below sees
    # a warm one — the two disagree and the gate fails for the wrong reason.
    _embedder_ready()
    assert _CORPUS.exists(), (
        f"{_CORPUS.name} missing — regenerate with "
        "`python -m app.eval.live_corpus_gen`")
    return run_corpus(_CORPUS)


def test_corpus_is_large_enough_to_be_meaningful(report):
    # Below ~1000 rows a single utterance moves F1 by more than the effect sizes
    # being measured, and the metrics stop being evidence.
    assert report["total"] >= 4000


def test_recall_a_question_is_almost_never_missed(report):
    """A missed question is SILENCE in a live interview."""
    floor = 0.98 if _embedder_ready() else 0.95
    assert report["recall"] >= floor, (
        f"recall {report['recall']} below {floor} — "
        f"{report['counts']['fn']} questions would get no answer")


def test_false_answer_rate_stays_under_the_target(report):
    """Answering a non-question talks OVER the interviewer. The harness's own
    stated target is < 5%."""
    assert report["false_answer_rate"] < 0.05, (
        f"false-answer rate {report['false_answer_rate']} — "
        f"{report['counts']['fp']} non-questions would be answered")


def test_f1_holds(report):
    floor = 0.98 if _embedder_ready() else 0.95
    assert report["f1"] >= floor


def test_fast_path_covers_almost_every_question(report):
    """Latency gate. A question the deterministic path cannot confirm pays an
    extra LLM round-trip before the answer even starts generating. This was
    0.63 — a third of all questions — until clause-level interrogatives and
    subject-auxiliary inversion were treated as the grammatical certainties
    they are."""
    assert report["fast_path_coverage"] >= 0.90, (
        f"fast-path coverage {report['fast_path_coverage']} — the remainder "
        "pay an extra detection round-trip")


def test_multi_question_recall_holds(report):
    assert report["multi_question_recall"] >= 0.85


@pytest.mark.parametrize("text", [
    # Every one of these was MISSED before the clause/inversion/imperative work,
    # and each is an ordinary thing an interviewer says. Kept as named cases so
    # a regression names the shape it broke rather than moving an aggregate.
    "Is Kafka something you have worked with",
    "Does it scale horizontally",
    "Should every team adopt microservices",
    "Have you had to tune the JVM garbage collector before",
    "Say you inherit a system using Docker, where do you start",
    "Suppose traffic doubles overnight, how would Redis hold up",
    "Talk me through your understanding of database indexing",
    "Take me through the internals of a hashmap",
    "Write a function that validates an email address",
    "Implement consistent hashing from scratch",
    "Rate your comfort with SQL",
    "And what about connection pooling",
    "How do you debug an issue with write-ahead logging",
    "Tell me about blue-green deployment",
])
def test_unpunctuated_questions_are_detected(text):
    """STT drops the terminal '?' constantly, so these arrive bare."""
    from app.question_detection.classifier import heuristic_classify
    assert heuristic_classify(text).is_question, f"missed: {text!r}"


@pytest.mark.parametrize("text", [
    # Interviewer housekeeping. Answering any of these talks OVER them.
    "Give me one moment, my screen froze",
    "Let me share my screen",
    "Okay, that makes sense",
    "Alright, switching gears",
    "We have three sections planned for this conversation",
])
def test_interviewer_housekeeping_takes_no_turn(text):
    from app.question_detection.classifier import heuristic_classify
    if not _embedder_ready():
        pytest.skip("semantic floor-holding veto needs the embedder")
    assert not heuristic_classify(text).is_question, f"false answer: {text!r}"


# ── Interviewer explains, then asks (the "did you handle this?" scenario) ────
#
# A real interviewer frequently answers their own question — "so the way it
# works is…" — and then asks a follow-up. Two distinct failures are possible and
# both are bad: answering the EXPLANATION talks over them, and missing the
# FOLLOW-UP leaves the candidate silent. These pin both directions on the same
# conversational turn.

@pytest.mark.parametrize("text", [
    "So Kafka keeps ordering per partition, that is the key idea",
    "Right, so the way it works is the broker appends to a log",
    "Basically consumers track their own offset",
    "Let me explain how our retry logic works",
    "I'll walk you through what we do here",
    "The reason we moved off it was operational cost",
])
def test_interviewer_explaining_is_not_answered(text):
    """Answering here talks OVER the interviewer mid-sentence."""
    from app.question_detection.classifier import heuristic_classify
    if not _embedder_ready():
        pytest.skip("the floor-holding veto needs the embedder")
    assert not heuristic_classify(text).is_question, f"false answer: {text!r}"


@pytest.mark.parametrize("text", [
    "And how would you handle a consumer lagging behind",
    "So given that, what would you change",
    "Does that make sense to you",
    "Now what about exactly-once semantics",
    "Okay so with that in mind, how would you design it",
    "But why would you pick that over the alternative",
])
def test_the_follow_up_after_an_explanation_is_detected(text):
    """The question that FOLLOWS the explanation is the one to answer. Missing
    it is silence at exactly the moment the candidate is expected to speak."""
    from app.question_detection.classifier import heuristic_classify
    assert heuristic_classify(text).is_question, f"missed follow-up: {text!r}"
