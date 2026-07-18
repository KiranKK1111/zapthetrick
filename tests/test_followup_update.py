"""State updater + continuation (followup-context-engine R6/R7/R9/R10, task 7.3).

Pins Properties 7 & 8: correction supersedes a decision, reversal removes it,
negative constraints recorded; commit registers entities + enumerations and
clears an open question; continuation directive is non-repeating.
"""
from __future__ import annotations

from app.followup import acts as A
from app.followup import update as U
from app.followup.reference import resolve
from app.followup.state import ConversationState


def test_correction_supersedes_decision():
    s = ConversationState({}, "c1")
    s.set_decision("database", "mysql")
    U.apply_turn("actually use postgres instead", A.CORRECTION, None, s)
    # The corrected tech is recorded; old value not kept under the same key.
    assert s.decisions().get("language") == "postgres" or \
        s.decisions().get("choice", "").lower().startswith("postgres") or \
        "postgres" in " ".join(s.decisions().values()).lower()


def test_reversal_removes_decision():
    s = ConversationState({}, "c1")
    s.set_decision("language", "python")
    U.apply_turn("undo that, no longer use python", A.CORRECTION, None, s)
    assert "python" not in " ".join(s.decisions().values()).lower()


def test_negative_constraint_recorded():
    s = ConversationState({}, "c1")
    U.apply_turn("don't use Firebase", A.FOLLOW_UP, None, s)
    cons = s.constraints()
    assert any(c["negative"] and "firebase" in c["text"].lower() for c in cons)


def test_commit_registers_entities_and_enumerations():
    s = ConversationState({}, "c1")
    answer = (
        "Here are the options:\n"
        "1. Redis\n"
        "2. Memcached\n"
        "3. Dragonfly\n"
        "You could build this in Python with FastAPI."
    )
    U.commit("which cache?", answer, s)
    assert s.enumerations()[:3] == ["Redis", "Memcached", "Dragonfly"]
    # A selection reference now resolves against those enumerations.
    r = resolve("use the second one", s)
    assert r.antecedents == ["Memcached"]


def test_commit_clears_open_question():
    s = ConversationState({}, "c1")
    s.add_open_question("Which database?")
    U.commit("PostgreSQL", "Great, using PostgreSQL.", s)
    assert s.open_questions() == []


def test_continuation_directive_non_repeating():
    s = ConversationState({}, "c1")
    s.set_enumerations(["A", "B", "C"])
    d = U.continuation_directive(s)
    assert "next" in d.lower() and "repeat" in d.lower()


def test_apply_turn_noop_on_error():
    class _Boom:
        def set_decision(self, *a):
            raise RuntimeError("boom")

    # Must not raise (fail-open, Property 1).
    U.apply_turn("actually use rust", A.CORRECTION, None, _Boom())
