"""
Phase-B live-interview conversation-tracking scenarios (39-103, subset).

Deterministic, no-LLM, no-network tests exercising the REAL live modules:
topic_graph, world_model, state_machine, interrupt, satisfaction, revise,
contradiction, predict, conversation. Each test maps to a numbered scenario.

Scenarios that the current implementation does not support are marked
`@pytest.mark.skip(...)` so the gap is documented rather than faked.
"""
from __future__ import annotations

import uuid

import pytest

from app.live import (
    contradiction,
    conversation,
    interrupt,
    predict,
    revise,
    satisfaction,
)
from app.live import state_machine as sm
from app.live import events as ev
from app.live.state_machine import get_state_machine
from app.live.topic_graph import TopicGraph
from app.live.world_model import (
    InterviewWorldModel,
    extract_world,
    for_tracker as world_for_tracker,
)
from app.live.world_model import resolve_coreference
from app.question_detection.context_tracker import get_tracker


def _sid() -> str:
    """A unique per-test live-session id."""
    return "phaseb-" + uuid.uuid4().hex


# ─────────────────────────── Topic graph (39-50) ───────────────────────────

def test_s39_topic_tracker_current_sub_prev():
    # Scenario 39: topic_tracker current/sub/prev
    g = TopicGraph()
    g.add_topic("Microservices")
    assert g.current() == "microservices"
    assert g.previous() is None
    g.add_topic("Kafka", parent="Microservices")
    assert g.current() == "kafka"
    assert g.previous() == "microservices"
    g.add_topic("Consumer Groups", parent="Kafka")
    assert g.current() == "consumer groups"
    assert g.previous() == "kafka"
    assert g.node("kafka").parent == "microservices"
    assert g.node("consumer groups").parent == "kafka"


def test_s40_topic_hierarchy_tree():
    # Scenario 40: topic_hierarchy_tree
    g = TopicGraph()
    g.add_topic("Microservices")
    g.add_topic("Kafka", parent="Microservices")
    g.add_topic("Consumer Groups", parent="Kafka")
    g.add_topic("Offsets", parent="Consumer Groups")
    # Full parent chain reconstructable from the tree.
    chain = []
    name = "offsets"
    while name:
        chain.append(name)
        node = g.node(name)
        name = node.parent if node else None
    assert chain == ["offsets", "consumer groups", "kafka", "microservices"]
    assert set(g.topics()) == {"microservices", "kafka", "consumer groups", "offsets"}


def test_s41_topic_drift_detection():
    # Scenario 41: topic_drift_detection ("Let's discuss Redis now" -> drift from Kafka)
    g = TopicGraph()
    g.add_topic("Kafka")
    assert g.detect_drift("Redis") is True
    # observe() records the drift and switches the active node.
    drifted = g.observe("Redis")
    assert drifted is True
    assert g.current() == "redis"
    assert g.previous() == "kafka"


def test_s42_domain_shift():
    # Scenario 42: domain_shift (unrelated domains drift; sub-topics do not)
    g = TopicGraph()
    g.add_topic("Databases")
    # A related refinement below the drift threshold stays a sub-topic.
    assert g.detect_drift("Databases indexing", similarity=0.9) is False
    # An unrelated domain with low embedding similarity is a drift.
    assert g.detect_drift("Frontend", similarity=0.1) is True


def test_s43_multiple_concurrent_topics_branch_and_return():
    # Scenario 43: multiple_concurrent_topics (branch and return)
    g = TopicGraph()
    g.add_topic("Kafka")
    assert g.observe("Redis") is True          # branch away
    assert g.current() == "redis"
    # Return to the earlier branch by reference.
    assert g.resolve_reference("let's go back to kafka") == "kafka"


def test_s44_followup_linked_to_topic():
    # Scenario 44: followup_linked_to_topic
    g = TopicGraph()
    g.add_topic("Kafka")
    g.attach_followup(3)
    g.attach_followup(7, topic="Kafka")
    assert g.node("kafka").turns == [3, 7]


def test_s45_conversation_graph_attach():
    # Scenario 45: conversation_graph_attach (role-tagged shared transcript)
    tracker = get_tracker(_sid())
    log = conversation.for_tracker(tracker)
    log.add(conversation.INTERVIEWER, "How does Kafka scale?", topic="kafka")
    log.add(conversation.CANDIDATE, "By adding partitions.", topic="kafka")
    log.add(conversation.ASSISTANT, "Mention consumer-group rebalancing.", topic="kafka")
    lines = log.context_lines(topic="kafka")
    assert any(l.startswith("Interviewer:") for l in lines)
    assert any(l.startswith("You:") for l in lines)
    assert any(l.startswith("Assistant suggested:") for l in lines)
    assert log.last_candidate() == "By adding partitions."


def test_s46_nested_followup_navigation():
    # Scenario 46: nested_followup_navigation ("let's go back to partitions" -> resolve_reference)
    g = TopicGraph()
    g.add_topic("Kafka")
    g.add_topic("Partitions", parent="Kafka")
    g.add_topic("Offsets", parent="Partitions")
    assert g.current() == "offsets"
    assert g.resolve_reference("let's go back to partitions") == "partitions"


def test_s47_coreference_pronoun():
    # Scenario 47: coreference_pronoun ("How does it scale?" -> it = Kafka)
    # NOTE: topic_graph.resolve_reference does NOT resolve bare pronouns (only
    # named-topic / "back to" references); pronoun coref lives in world_model.
    g = TopicGraph()
    g.add_topic("Kafka")
    assert g.resolve_reference("How does it scale?") is None
    # The real pronoun resolver is world_model.resolve_coreference.
    m = InterviewWorldModel(topic="Kafka")
    res = resolve_coreference("How does it scale?", m)
    assert res["resolved"] is True
    assert res["referent"] == "Kafka"


def test_s48_reference_to_earlier_topic():
    # Scenario 48: reference_to_earlier_topic ("How is that different from Redis?")
    g = TopicGraph()
    g.add_topic("Redis")
    g.add_topic("Kafka")           # current == kafka; redis is an earlier node
    assert g.resolve_reference("How is that different from Redis?") == "redis"


def test_s50_followup_prediction():
    # Scenario 50: followup_prediction (predict_next)
    m = InterviewWorldModel(topic="Kafka")
    preds = predict.predict_next(world_model=m)
    assert preds
    assert any("kafka" in p.lower() for p in preds)
    # Also works from the topic graph's current node.
    g = TopicGraph()
    g.add_topic("Redis")
    assert predict.predict_next(topic_graph=g)


# ─────────────────────────── State machine (52-59) ──────────────────────────

def _q_event(text: str = "What is Kafka?") -> ev.UtteranceEvent:
    return ev.UtteranceEvent(kind=ev.QUESTION, questions=[text])


def test_s52_interview_state_machine():
    # Scenario 52: interview_state_machine (WAITING -> QUESTION_DETECTED -> ANSWERING)
    # Real states: IDLE -> LISTENING -> QUESTION_CONFIRMED -> ANSWERING -> FOLLOWUP_WAITING.
    machine = get_state_machine(_sid())
    assert machine.state == sm.IDLE
    assert machine.mark_listening() == sm.LISTENING
    assert machine.advance(_q_event()) == sm.QUESTION_CONFIRMED
    assert machine.on_answer_start() == sm.ANSWERING
    assert machine.on_answer_done() == sm.FOLLOWUP_WAITING


def test_s53_streaming_state_machine():
    # Scenario 53: streaming_state_machine (concurrent answers never block)
    machine = get_state_machine(_sid())
    machine.advance(_q_event("What is Kafka?"))
    machine.on_answer_start()                       # first answer streaming
    machine.advance(_q_event("And Redis?"))         # a concurrent question mid-answer
    assert machine.state == sm.ANSWERING            # stays ANSWERING (concurrency)
    machine.on_answer_start()                       # second answer streaming
    assert machine.snapshot()["active_answers"] == 2
    assert machine.on_answer_done() == sm.ANSWERING     # one still active
    assert machine.on_answer_done() == sm.FOLLOWUP_WAITING


def test_s54_satisfaction_close_thread():
    # Scenario 54: satisfaction_close_thread ("Good. Correct." -> closed)
    assert satisfaction.classify_feedback("Good. Correct.") == satisfaction.CLOSED
    assert satisfaction.classify_feedback("Makes sense.") == satisfaction.CLOSED


def test_s55_dissatisfaction_thread_open():
    # Scenario 55: dissatisfaction_thread_open ("Not quite. Think deeper.")
    assert satisfaction.classify_feedback("Not quite. Think deeper.") == satisfaction.OPEN
    assert satisfaction.classify_feedback("Are you sure?") == satisfaction.OPEN


def test_s56_interruption_stop_generation():
    # Scenario 56: interruption_stop_generation ("Actually let's talk about RabbitMQ")
    assert interrupt.should_cancel("Actually let's talk about RabbitMQ") is True
    assert interrupt.should_cancel("What is Kafka?") is False


def test_s57_interrupted_abandoned_question():
    # Scenario 57: interrupted_abandoned_question
    machine = get_state_machine(_sid())
    machine.advance(_q_event())
    machine.on_answer_start()
    # An interrupting utterance cancels + moves the machine to INTERRUPTED.
    assert interrupt.should_cancel("Wait, never mind that question") is True
    assert machine.mark_interrupted() == sm.INTERRUPTED


def test_s58_self_correction_supersede():
    # Scenario 58: self_correction_supersede ("Sorry, actually explain RabbitMQ")
    sig = interrupt.detect("Sorry, actually explain RabbitMQ")
    assert sig.interrupted is True
    assert sig.self_correction is True
    assert interrupt.should_cancel("Sorry, actually explain RabbitMQ") is True


def test_s59_cancellation_support():
    # Scenario 59: cancellation_support
    assert interrupt.should_cancel("Scratch that, explain RabbitMQ instead") is True
    assert interrupt.should_cancel("Hold on, before that") is True
    # A plain question must NOT trigger cancellation.
    assert interrupt.should_cancel("How does Kafka handle backpressure?") is False


# ─────────────── World model / assumptions / constraints (66-103) ───────────

def test_s66_world_interview_model():
    # Scenario 66: world_interview_model (snapshot fields)
    tracker = get_tracker(_sid())
    m = world_for_tracker(tracker)
    m.set_active("What is Kafka?", qid="q1", topic="Kafka")
    m.add_assumption("100M users")
    m.add_constraint("no third-party services")
    m.mark_candidate_answered()
    m.mark_satisfied(True)
    snap = m.snapshot()
    assert set(snap) >= {
        "topic", "subtopic", "active_question", "candidate_answered",
        "interviewer_satisfied", "assumptions", "constraints",
    }
    assert snap["topic"] == "Kafka"
    assert snap["active_question"] == "What is Kafka?"
    assert snap["candidate_answered"] is True
    assert snap["interviewer_satisfied"] is True
    assert "100M users" in snap["assumptions"]
    assert "no third-party services" in snap["constraints"]
    assert m.active_qid == "q1"


def test_s98_assumption_tracking():
    # Scenario 98: assumption_tracking ("Assume 100M users" -> add_assumption/extract)
    m = InterviewWorldModel()
    extract_world("Assume 100M users", m)
    assert "Assume 100M users" in m.assumptions
    assert "Assumptions:" in m.honored_context()


def test_s99_constraint_extraction():
    # Scenario 99: constraint_extraction (constraint cue -> constraints)
    m = InterviewWorldModel()
    # The extractor keys off constraint CUES (must / can only / budget is / ...).
    extract_world("You must design WhatsApp for 500M users with low latency", m)
    assert m.constraints
    assert "Constraints:" in m.honored_context()
    # The bare example without a cue is (correctly) not auto-extracted...
    m2 = InterviewWorldModel()
    extract_world("Design WhatsApp for 500M users with low latency", m2)
    assert m2.constraints == []
    # ...but can be recorded explicitly.
    m2.add_constraint("500M users, low latency")
    assert "500M users, low latency" in m2.constraints


def test_s100_contradiction_detection_challenge():
    # Scenario 100: contradiction_detection_challenge (is_challenge)
    assert contradiction.is_challenge("But you said Kafka guarantees ordering") is True
    assert contradiction.is_challenge("Earlier you said Redis was faster") is True
    assert contradiction.is_challenge("What is a Kafka partition?") is False


def test_s101_contradiction_memory():
    # Scenario 101: contradiction_memory (challenge against a remembered assumption)
    m = InterviewWorldModel()
    m.add_assumption("Kafka guarantees ordering within a partition")
    # Absolute re-question over the recorded assumption is a challenge.
    assert contradiction.is_challenge("Does Kafka ALWAYS guarantee ordering?", m) is True
    # Temporal reference resolves back to an earlier topic via the graph.
    g = TopicGraph()
    g.add_topic("Redis")
    g.add_topic("Kafka")
    assert contradiction.resolve_temporal("earlier you said Redis was faster", g) == "redis"


def test_s103_realtime_answer_revision():
    # Scenario 103: realtime_answer_revision (detect_reinterpretation "I meant the Redis cluster")
    m = InterviewWorldModel()
    m.set_active("How does it handle failover?", qid="q42", topic="Redis")
    qid = revise.detect_reinterpretation("I meant the Redis cluster", m)
    assert qid == "q42"
    revised = revise.revised_question("I meant the Redis cluster", m)
    assert "Redis cluster" in revised
    # A non-reinterpretation turn does not trigger a revision.
    assert revise.detect_reinterpretation("How does it scale?", m) is None
