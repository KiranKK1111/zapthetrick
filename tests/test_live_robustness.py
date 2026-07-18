"""Robustness signals + event log
(live-conversational-intelligence R11-R14; tasks 10.2, 11.2).

Pins Properties 11-14: conservative transcript repair, ensemble FP reduction,
interruption / self-correction / satisfaction detection, uncertainty
propagation, bounded event log + interviewer-style adaptation.
"""
from __future__ import annotations

from app.live import (
    ensemble,
    eventlog,
    interrupt,
    repair,
    satisfaction,
    style,
    uncertainty,
)


# ---- transcript repair (conservative) ----------------------------------
def test_repair_fixes_garbled_domain_term():
    out = repair.repair("explain kuberentes ingress", vocab=["ingress"])
    assert "kubernetes" in out.lower()


def test_repair_keeps_real_word():
    # "cohesion" is a real word — must NOT be swapped for a domain term.
    out = repair.repair("explain cohesion and coupling")
    assert "cohesion" in out.lower()
    assert "coupling" in out.lower()


def test_repair_does_not_touch_correct_terms():
    assert repair.repair("what is kafka") == "what is kafka"


def test_repair_failopen_empty():
    assert repair.repair("") == ""


# ---- ensemble detection -------------------------------------------------
def test_ensemble_agent_question_stays_question():
    d = ensemble.decide(agent_is_q=True, agent_conf=0.85,
                        heuristic_is_q=True, heuristic_conf=0.7, prosody_score=0.6)
    assert d.is_question is True
    assert d.score >= 0.5


def test_ensemble_reduces_false_positive_when_agent_says_no():
    # Heuristic alone would say "question"; the agent (dominant) says no →
    # the ensemble does NOT produce a false-positive question.
    d = ensemble.decide(agent_is_q=False, agent_conf=0.85,
                        heuristic_is_q=True, heuristic_conf=0.7, prosody_score=0.2)
    assert d.is_question is False


def test_ensemble_never_drops_confident_agent_question():
    # Even with weak heuristic + low prosody, an agent-confirmed question stays.
    d = ensemble.decide(agent_is_q=True, agent_conf=0.85,
                        heuristic_is_q=False, heuristic_conf=0.2, prosody_score=0.1)
    assert d.is_question is True


# ---- interruption / self-correction ------------------------------------
def test_interruption_detected():
    assert interrupt.should_cancel("actually, before that, how many partitions?") is True
    assert interrupt.detect("sorry, I meant the Redis cluster").self_correction is True


def test_no_interruption_on_plain_question():
    assert interrupt.should_cancel("How does Kafka handle ordering?") is False


# ---- satisfaction -------------------------------------------------------
def test_satisfaction_closed_and_open():
    assert satisfaction.classify_feedback("Great, makes sense") == satisfaction.CLOSED
    assert satisfaction.classify_feedback("not quite, think deeper") == satisfaction.OPEN
    assert satisfaction.classify_feedback("How would you scale it?") is None


# ---- uncertainty propagation -------------------------------------------
def test_uncertainty_lowers_on_poor_stt():
    high = uncertainty.propagate(0.9, stt_conf=0.95)
    low = uncertainty.propagate(0.9, stt_conf=0.1)
    assert low < high
    assert low <= 0.55


def test_uncertainty_no_signals_unchanged():
    assert uncertainty.propagate(0.8) == 0.8


# ---- interviewer style --------------------------------------------------
def test_style_rapid_fire_lowers_threshold():
    s = style.InterviewerStyle()
    for _ in range(5):
        s.observe(question="scale it?", is_followup=True)
    assert s.label() == style.RAPID_FIRE
    assert s.threshold_adjustment() < 0


def test_style_recent_lens_bounded():
    s = style.InterviewerStyle()
    for i in range(50):
        s.observe(question="a b c")
    assert len(s._recent_lens) <= 20


# ---- event log ----------------------------------------------------------
def test_event_log_bounded_and_appends():
    log = eventlog.EventLog(maxlen=10)
    for i in range(25):
        log.append("token", {"i": i})
    assert len(log) == 10
    assert log.events()[-1]["data"]["i"] == 24


def test_event_log_registry_per_session():
    a = eventlog.get_log("s1")
    assert a is eventlog.get_log("s1")
    assert a is not eventlog.get_log("s2")
    eventlog.forget_session("s1")
    assert eventlog.get_log("s1") is not a
