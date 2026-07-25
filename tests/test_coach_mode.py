"""Tests for coach mode — role-flip mock interview (vNext §9.6, Stage 11 C)."""
from __future__ import annotations

from dataclasses import dataclass

import app.live.coach_mode as C


@dataclass
class _Contract:
    sections: tuple = ("context", "approach & tradeoffs", "impact")
    must_include: tuple = ("a measurable result",)
    avoid: tuple = ("org-level impact",)
    max_seconds: int = 45


_QS = [
    {"question": "Tell me about yourself", "category": "introduction"},
    {"question": "How have you used Kafka?", "category": "technical"},
    {"question": "Design a rate limiter", "category": "design"},
    {"question": "Walk me through a hard bug", "category": "behavioural"},
]


# ---- question-arc selector ------------------------------------------------
def test_arc_ascending_pressure():
    arc = C.build_arc(_QS, band="senior")
    assert arc[0].phase == C.WARM_UP
    assert arc[0].category in ("introduction", "behavioural")   # warm-up leads
    phases = [a.phase for a in arc]
    assert C.CORE in phases and C.DEEP in phases


def test_arc_orders_and_indexes():
    arc = C.build_arc(_QS)
    assert [a.order for a in arc] == list(range(len(arc)))


def test_arc_accepts_strings():
    arc = C.build_arc(["Q1", "Q2"])
    assert len(arc) == 2 and arc[0].question == "Q1"


def test_arc_respects_max_len():
    arc = C.build_arc([{"question": f"q{i}"} for i in range(20)], max_len=4)
    assert len(arc) == 4


def test_arc_empty():
    assert C.build_arc([]) == []
    assert C.build_arc(None) == []


# ---- answer scoring -------------------------------------------------------
def test_strong_answer_scores_high():
    ans = ("The context was a slow API. My approach weighed caching tradeoffs "
           "against consistency. The impact was a measurable result: 40% faster.")
    s = C.score_answer(ans, _Contract())
    assert s.score >= 90 and s.coverage == 1.0 and s.included and s.disciplined


def test_weak_answer_scores_low_with_gaps():
    s = C.score_answer("I made it faster.", _Contract())
    assert s.score < 40
    assert any("missing" in g for g in s.gaps)


def test_over_claim_flagged_and_penalized():
    ans = ("context here. approach & tradeoffs. impact: a measurable result. "
           "I drove org-level impact across the whole company.")
    s = C.score_answer(ans, _Contract())
    assert not s.disciplined
    assert any("over-claim" in g for g in s.gaps)


def test_missing_required_element():
    ans = "context. approach & tradeoffs. impact happened."
    s = C.score_answer(ans, _Contract())
    assert not s.included
    assert any("needs" in g for g in s.gaps)


def test_injected_judge_used():
    s = C.score_answer("x", _Contract(),
                       judge_fn=lambda a, c: {"score": 77, "coverage": 0.9,
                                              "included": True, "disciplined": True,
                                              "concise": True, "gaps": []})
    assert s.score == 77 and s.coverage == 0.9


def test_score_never_raises():
    # A null contract has no rubric → nothing to fail against (not a crash).
    s = C.score_answer(None, None)   # type: ignore[arg-type]
    assert isinstance(s, C.AnswerScore)


def test_judge_error_falls_back_to_deterministic():
    ans = "context. approach & tradeoffs. impact: a measurable result."
    s = C.score_answer(ans, _Contract(), judge_fn=lambda a, c: 1 / 0)
    assert isinstance(s, C.AnswerScore) and s.coverage == 1.0   # deterministic ran


# ---- follow-up decision ---------------------------------------------------
def _score(score, disciplined=True):
    return C.AnswerScore(score, 1.0, True, disciplined, True, [])


def test_weak_drills():
    assert C.follow_up_decision(_score(40)).action == C.DRILL


def test_strong_moves_on():
    assert C.follow_up_decision(_score(90)).action == C.MOVE_ON


def test_solid_advances():
    assert C.follow_up_decision(_score(70)).action == C.ADVANCE


def test_over_claim_always_drills():
    assert C.follow_up_decision(_score(95, disciplined=False)).action == C.DRILL


# ---- coach session --------------------------------------------------------
def test_session_advances_on_strong_stays_on_drill():
    sess = C.CoachSession(band="senior", arc=C.build_arc(_QS))
    assert sess.current() is not None
    step = sess.record(_score(90))                # move_on
    assert step.action == C.MOVE_ON and sess.pos == 1
    step = sess.record(_score(30))                # drill → stay
    assert step.action == C.DRILL and sess.pos == 1


def test_session_debrief_summary():
    sess = C.CoachSession(band="senior", arc=C.build_arc(_QS))
    sess.record(C.AnswerScore(90, 1.0, True, True, True, []))
    sess.record(C.AnswerScore(30, 0.3, False, True, True, ["missing: impact"]))
    d = sess.debrief()
    assert d["answered"] == 2 and d["average"] == 60
    assert "missing: impact" in d["focus_areas"]
    assert len(d["per_question"]) == 2


def test_session_done():
    sess = C.CoachSession(arc=C.build_arc(["q1"]))
    assert not sess.done()
    sess.record(_score(90))
    assert sess.done()


def test_empty_session_debrief():
    d = C.CoachSession().debrief()
    assert d["answered"] == 0 and d["focus_areas"] == []
