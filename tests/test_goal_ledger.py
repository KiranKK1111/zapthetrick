"""Tests for the cross-turn goal ledger + conversation state
(advanced-intent-reasoning Phase 3). Dict-backed — no DB, no LLM."""
from __future__ import annotations

from app.clarify.goal_ledger import (
    DISCOVERY,
    EXECUTION,
    PLANNING,
    REVIEW,
    GoalLedger,
    classify_state,
    threshold_for,
)


class TestGoalLedger:
    def test_accumulates_and_suppresses_slots(self):
        prefs: dict = {}
        led = GoalLedger(prefs, "c1")
        led.observe("build a react app", "")
        slots = led.confirmed_slots()
        assert slots.get("framework") == "react"
        assert led.has_tech()

    def test_slots_persist_across_turns(self):
        prefs: dict = {}
        GoalLedger(prefs, "c1").observe("let's use python", "")
        # New ledger over the same root for a later turn in the same conv.
        led2 = GoalLedger(prefs, "c1")
        assert led2.confirmed_slots().get("language") == "python"

    def test_does_not_overwrite_earlier_decision(self):
        prefs: dict = {}
        led = GoalLedger(prefs, "c1")
        led.observe("use python", "")
        led.observe("also mentions java somewhere", "")
        # First-decided language wins (we don't clobber a confirmed slot).
        assert led.confirmed_slots()["language"] == "python"

    def test_record_choice(self):
        prefs: dict = {}
        led = GoalLedger(prefs, "c1")
        led.record_choice("Platform", "web")
        assert led.confirmed_slots()["platform"] == "web"

    def test_scoped_per_conversation(self):
        prefs: dict = {}
        GoalLedger(prefs, "c1").observe("use python", "")
        led2 = GoalLedger(prefs, "c2")
        assert led2.confirmed_slots() == {}

    def test_no_conversation_is_inert(self):
        prefs: dict = {}
        led = GoalLedger(prefs, None)
        led.observe("use python", "")
        assert led.confirmed_slots() == {}

    def test_shares_root_with_siblings(self):
        prefs = {"clarify": {"durable": {}}, "clarify_outcomes": {"ring": []}}
        GoalLedger(prefs, "c1").observe("use go", "")
        assert "clarify_ledger" in prefs
        assert "clarify" in prefs and "clarify_outcomes" in prefs


class TestConversationState:
    def test_discovery_default(self):
        assert classify_state("", "tell me about options") == DISCOVERY

    def test_planning(self):
        assert classify_state("", "how should I structure the schema") == PLANNING

    def test_execution(self):
        assert classify_state("", "now build the login screen") == EXECUTION

    def test_review_wins_over_execution(self):
        # "fix" (review) should win even if "build" appears.
        assert classify_state("build the app", "fix the failing build") == REVIEW

    def test_threshold_ordering(self):
        # Discovery tolerates more asking (higher answer band); execution asks
        # least (lower answer band → answers more).
        assert threshold_for(DISCOVERY) > threshold_for(EXECUTION)
        assert 0.0 < threshold_for(EXECUTION) < threshold_for(PLANNING) <= 1.0
        assert threshold_for("nonsense") == 0.90
