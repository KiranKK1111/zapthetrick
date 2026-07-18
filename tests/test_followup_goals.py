"""Open-loop + goal-evolution tracking (followup-context-engine R9/R10, task 10.3).

Pins Property 10: open questions add/clear; a goal shift updates the current
goal while retaining the initial one; a completed goal is marked so continuation
advances.
"""
from __future__ import annotations

from app.followup import acts as A
from app.followup import update as U
from app.followup.state import ConversationState


def test_open_question_add_and_clear():
    s = ConversationState({}, "c1")
    s.add_open_question("Which auth provider?")
    assert "Which auth provider?" in s.open_questions()
    # The user answers it → commit clears the open loop (R9.3).
    U.commit("Use Auth0", "Configured Auth0.", s)
    assert s.open_questions() == []


def test_goal_shift_retains_initial_goal():
    s = ConversationState({}, "c1")
    s.observe("build a Flutter chat app with realtime sync", "")
    initial = s.goal_initial()
    assert initial is not None
    # A genuine new topic mid-conversation shifts the current goal.
    U.apply_turn("now design a billing and subscription system for it",
                 A.NEW_TOPIC, None, s)
    assert s.goal_initial() == initial          # initial retained (R10.1)
    assert s.goal() != initial                  # current goal advanced


def test_goal_completion_marked():
    s = ConversationState({}, "c1")
    s.set_goal("write the API spec")
    U.apply_turn("that's all, we're done here", A.APPROVAL, None, s)
    assert s.is_goal_complete() is True          # (R10.2)


def test_thread_state_persists_per_conversation():
    """Resuming a conversation restores its goal/decisions/open-questions via the
    shared persisted root (R9.2 within-conversation)."""
    root: dict = {}
    s1 = ConversationState(root, "c1")
    s1.set_goal("migrate to microservices")
    s1.add_open_question("Which message bus?")
    # Later turn / relaunch loads the same root.
    s2 = ConversationState(root, "c1")
    assert s2.goal() == "migrate to microservices"
    assert "Which message bus?" in s2.open_questions()
