"""Conversational depth
(live-conversational-intelligence R28, R30-R34; tasks 23.3).

Pins Properties 28, 30, 31, 32, 33, 34: diarization roles/panel +
candidate-not-answered, world-model assumptions/constraints + bounding,
objective/depth estimation, answer-revision qid targeting, false-premise
correction, contradiction/temporal resolution.
"""
from __future__ import annotations

from app.live import contradiction, diarize, objective, premise, revise
from app.live.diarize import CANDIDATE, PRIMARY, SECONDARY, Diarizer
from app.live.topic_graph import TopicGraph
from app.live.world_model import InterviewWorldModel, extract_world


# ---- diarization -------------------------------------------------------
def test_diarize_candidate_vs_interviewer():
    d = Diarizer()
    role, _ = d.attribute(source="mic", text="I would shard the cache")
    assert role == CANDIDATE
    assert d.is_candidate(role) is True
    role2, _ = d.attribute(source="system_loopback", text="What is Kafka?")
    assert role2 == PRIMARY


def test_diarize_panel_handoff():
    d = Diarizer()
    d.attribute(source="system_loopback", text="My colleague will take over from here")
    role, _ = d.attribute(source="system_loopback", text="How do you test this?")
    assert role == SECONDARY
    assert d.panel_size() >= 1


# ---- world model -------------------------------------------------------
def test_world_model_assumptions_constraints():
    w = InterviewWorldModel()
    extract_world("Let's say we have 100 million users", w)
    extract_world("You cannot use any third-party services", w)
    assert any("100 million" in a for a in w.assumptions)
    assert any("third-party" in c for c in w.constraints)
    assert "Assumptions:" in w.honored_context()
    assert "Constraints:" in w.honored_context()


def test_world_model_bounded():
    w = InterviewWorldModel()
    for i in range(40):
        w.add_assumption(f"assumption {i}")
    assert len(w.assumptions) <= 24


def test_world_model_active_question_resets_flags():
    w = InterviewWorldModel()
    w.mark_candidate_answered()
    w.set_active("How would you scale it?", qid="q9", topic="kafka")
    assert w.active_qid == "q9"
    assert w.candidate_answered is False


# ---- objective + depth -------------------------------------------------
def test_objective_tradeoff_and_design():
    obj, _ = objective.estimate("What is the CAP theorem?")
    assert obj == objective.TRADEOFF
    obj2, depth2 = objective.estimate("Design a system that scales to 1B users")
    assert obj2 == objective.DESIGN_ABILITY
    assert depth2 == objective.ARCHITECTURE


def test_objective_depth_from_difficulty():
    _, depth = objective.estimate("Explain consensus", difficulty="expert")
    assert depth == objective.INTERNALS
    assert objective.directive(objective.KNOWLEDGE, objective.DEFINITION)


# ---- answer revision ---------------------------------------------------
def test_revision_targets_prior_qid():
    w = InterviewWorldModel()
    w.set_active("How would you scale it?", qid="q1")
    assert revise.detect_reinterpretation("I meant the Redis cluster", w) == "q1"
    assert revise.detect_reinterpretation("How do partitions work?", w) is None
    rq = revise.revised_question("I meant the Redis cluster", w)
    assert "Redis" in rq


# ---- false premise -----------------------------------------------------
def test_false_premise_flagged_and_directive():
    p = premise.check_premise("Kafka stores data only in memory, right?")
    assert p.false_premise is True
    assert "premise" in premise.directive(p).lower()


def test_no_false_premise_on_plain_question():
    p = premise.check_premise("How does Kafka persist messages?")
    assert p.false_premise is False
    assert premise.directive(p) == ""


# ---- contradiction + temporal -----------------------------------------
def test_is_challenge_detects_callback():
    assert contradiction.is_challenge("But you said it always guarantees ordering") is True
    assert contradiction.is_challenge("What is a consumer group?") is False


def test_challenge_on_assumption_absolute():
    w = InterviewWorldModel()
    w.add_assumption("Kafka guarantees ordering")
    assert contradiction.is_challenge("Does Kafka always guarantee ordering?", w) is True


def test_resolve_temporal_against_topic_graph():
    g = TopicGraph()
    g.observe("partitions")
    g.observe("consumer groups")
    assert contradiction.resolve_temporal("let's go back to partitions", g) == "partitions"
    assert contradiction.resolve_temporal("what is a broker?", g) is None
