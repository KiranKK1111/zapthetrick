"""Follow-up prompt rewriting (followup-context-engine R5, task 6.3).

Pins Property 6: a confident rewrite is explicit + self-contained and invents no
new decisions/constraints; low confidence / no antecedent → the original turn.
"""
from __future__ import annotations

from app.followup import acts as A
from app.followup.rewrite import rewrite
from app.followup.reference import resolve
from app.followup.state import ConversationState


def _state():
    s = ConversationState({}, "c1")
    s.set_goal("a Flutter streaming architecture")
    s.add_entity("Flutter")
    return s


def test_followup_rewrite_is_explicit_and_self_contained():
    s = _state()
    res = resolve("make it faster", s)            # "it" → Flutter
    text, conf = rewrite("make it faster", A.FOLLOW_UP, res, s)
    assert conf >= 0.6
    assert text != "make it faster"
    assert "Flutter" in text                       # concrete antecedent injected


def test_rewrite_introduces_no_new_decisions():
    s = _state()
    res = resolve("make it better", s)
    text, _ = rewrite("make it better", A.FOLLOW_UP, res, s)
    # The rewrite must not invent a tech/decision the user never stated.
    for invented in ("python", "react", "postgres", "django"):
        assert invented not in text.lower()


def test_continuation_rewrite_resumes_without_repeating():
    s = _state()
    text, conf = rewrite("continue", A.CONTINUATION, resolve("continue", s), s)
    assert "without repeating" in text.lower() or "from where it ended" in text.lower()


def test_low_confidence_falls_back_to_original():
    # No goal, no entities → no antecedent → confidence 0 → original returned.
    s = ConversationState({}, "c2")
    res = resolve("make it better", s)
    text, conf = rewrite("make it better", A.FOLLOW_UP, res, s)
    assert text == "make it better"
    assert conf == 0.0


def test_non_followup_acts_return_original():
    s = _state()
    text, conf = rewrite("yes", A.APPROVAL, resolve("yes", s), s)
    assert text == "yes" and conf == 0.0
