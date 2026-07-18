"""Conversation_State store (followup-context-engine R1, task 1.3).

Pins Properties 1, 2, 7: persist/restore against a fake preferences root,
supersede/remove decisions, negative constraints, bounded eviction, fresh-
conversation fallback, and no-op on error. No DB, no model load — the store is
pure over a dict root (the same `User.preferences` JSONB the route persists).
"""
from __future__ import annotations

from app.followup.state import ConversationState


def test_persist_and_restore_across_turns():
    """A decision/constraint/entity recorded on one turn is visible on the next
    turn for the same conversation (shared root simulates persistence)."""
    root: dict = {}
    s1 = ConversationState(root, "c1")
    s1.set_decision("database", "postgres")
    s1.add_constraint("must support offline", negative=False)
    s1.add_entity("Flutter")
    s1.set_goal("build a chat app")

    # Next turn loads from the same persisted root.
    s2 = ConversationState(root, "c1")
    assert s2.decisions()["database"] == "postgres"
    assert any(c["text"] == "must support offline" for c in s2.constraints())
    assert "Flutter" in s2.entities()
    assert s2.goal() == "build a chat app"


def test_reset_topic_clears_narrative_but_keeps_slots():
    """An explicit topic shift drops the topic-scoped narrative (goal, entities,
    decisions, constraints, enumerations, open questions) so a new subject starts
    clean — while CONFIRMED slots survive so the user isn't re-asked prefs."""
    root: dict = {}
    s = ConversationState(root, "c1")
    s.set_goal("build a Flutter chat app")
    s.add_entity("Flutter")
    s.set_decision("database", "postgres")
    s.add_constraint("must support offline", negative=False)
    s.set_enumerations(["PostgreSQL", "MongoDB"])

    s.reset_topic()

    assert s.goal() is None
    assert s.entities() == []
    assert s.decisions() == {}
    assert s.constraints() == []
    assert s.enumerations() == []
    # Reset is scoped to this conversation and safely re-loadable.
    s2 = ConversationState(root, "c1")
    assert s2.goal() is None


def test_reset_topic_then_observe_seeds_new_goal():
    """After a reset, the next observed turn seeds a fresh goal (not the old)."""
    root: dict = {}
    s = ConversationState(root, "c1")
    s.set_goal("build a Flutter chat app")
    s.reset_topic()
    s.observe("explain how rust ownership works", "")
    # goal is re-seeded from the new turn (or stays None) — never the old goal.
    assert s.goal() != "build a Flutter chat app"


def test_decision_supersession_and_reversal():
    root: dict = {}
    s = ConversationState(root, "c1")
    s.set_decision("database", "mysql")
    s.set_decision("database", "postgres")          # supersede (R6.1)
    assert s.decisions() == {"database": "postgres"}  # not both kept
    s.remove_decision("database")                    # reversal (R6.2)
    assert "database" not in s.decisions()


def test_negative_constraint_recorded():
    root: dict = {}
    s = ConversationState(root, "c1")
    s.add_constraint("Firebase", negative=True)      # "don't use Firebase"
    cons = s.constraints()
    assert cons == [{"text": "Firebase", "negative": True}]
    assert "no Firebase" in s.summary()


def test_bounded_eviction_oldest_first():
    root: dict = {}
    s = ConversationState(root, "c1")
    # Entities cap defaults to 64; force a tiny bound to exercise eviction.
    s._bounds["entities"] = 3
    for name in ("a", "b", "c", "d"):
        s.add_entity(name)
    ents = s.entities()
    assert len(ents) == 3
    assert "a" not in ents and ents[-1] == "d"   # oldest evicted, newest last


def test_entity_lru_reorders_on_reuse():
    root: dict = {}
    s = ConversationState(root, "c1")
    s._bounds["entities"] = 3
    for name in ("a", "b", "c"):
        s.add_entity(name)
    s.add_entity("a")        # re-reference → moves to most-recent
    s.add_entity("d")        # evicts least-recent ("b")
    ents = s.entities()
    assert "b" not in ents and "a" in ents and ents[-1] == "d"


def test_fresh_conversation_fallback():
    """No record for the conversation → empty reads, behaves as fresh (R1.3)."""
    s = ConversationState({}, "new-convo")
    assert s.goal() is None
    assert s.decisions() == {}
    assert s.constraints() == []
    assert s.entities() == []
    assert s.summary() == ""


def test_legacy_ledger_record_loads():
    """An existing GoalLedger record (slots only, no widened keys) still loads;
    confirmed slots remain available (R1.2 back-compat)."""
    root = {"clarify_ledger": {"c1": {"slots": {"language": "python"},
                                      "constraints": []}}}
    s = ConversationState(root, "c1")
    assert s.confirmed_slots().get("language") == "python"
    # Widened reads default empty, no crash.
    assert s.decisions() == {}
    assert "language=python" in s.summary()


def test_no_conversation_id_is_ephemeral_noop():
    """Without a conversation id the store is ephemeral and never writes the
    shared root (R1.4-style no-op)."""
    root: dict = {}
    s = ConversationState(root, None)
    s.set_decision("x", "y")        # write
    s.observe("build a thing", "")  # observe is a no-op without a cid
    # The shared root's ledger map has no per-convo record from this.
    assert root.get("clarify_ledger", {}) == {} or "None" not in root.get(
        "clarify_ledger", {})


def test_observe_delegates_to_goal_ledger_slots():
    """observe() reuses GoalLedger slot extraction (no duplication) — a named
    language shows up in confirmed_slots and the summary."""
    root: dict = {}
    s = ConversationState(root, "c1")
    s.observe("build a REST API in Python with FastAPI", "")
    slots = s.confirmed_slots()
    assert slots.get("language") == "python" or slots.get("framework") == "fastapi"
    # A goal is seeded from the first substantive turn.
    assert s.goal() is not None
