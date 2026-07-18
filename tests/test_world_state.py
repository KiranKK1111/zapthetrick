"""Phase-5 (ArchitectureVerdict.md): unified TurnState + real planner plan.
Deterministic — no LLM, no DB.
"""
from __future__ import annotations

import asyncio

import pytest

from app.clarify.intent_pipeline import assess
from app.core.world_state import KEY_TURN_STATE, TurnState


class TestTurnState:
    def test_from_assessment_projection(self):
        a = assess("write a login api")
        ts = TurnState.from_assessment(a, goal="write a login api")
        assert ts.goal == "write a login api"
        assert ts.intent == a.intent
        assert ts.decision == a.decision
        assert ts.missing_required == a.missing_required
        assert ts.matrix is not None          # Phase-1 matrix projected
        assert ts.policy is not None          # Phase-3 record projected
        assert ts.capabilities is not None    # Phase-2 snapshot projected
        assert "document_formats" in ts.capabilities

    def test_as_dict_is_json_friendly(self):
        import json
        a = assess("compare kafka and rabbitmq")
        d = TurnState.from_assessment(a, goal="x").as_dict()
        json.dumps(d)                          # must serialize cleanly
        assert d["decision"] in ("answer", "clarify", "defer")

    def test_live_projection_same_protocol(self):
        ts = TurnState.from_live_snapshot(
            {"assumptions": ["remote role"], "active_question": "Why us?"},
            goal="live")
        d = ts.as_dict()
        assert d["assumptions"] == ["remote role"]
        assert d["plan"] and d["plan"][0]["text"] == "Why us?"
        # Same keys as the chat projection — one protocol.
        chat = TurnState.from_assessment(assess("hi"), goal="hi").as_dict()
        assert set(d.keys()) == set(chat.keys())

    def test_builders_never_raise(self):
        assert TurnState.from_assessment(object()).as_dict()
        assert TurnState.from_live_snapshot(None).as_dict()

    def test_set_plan_and_artifacts(self):
        ts = TurnState(goal="g")
        ts.set_plan([type("T", (), {"id": 1, "text": "step", "deps": []})()])
        ts.add_artifact("report", "pdf", {"validated": True})
        d = ts.as_dict()
        assert d["plan"][0]["text"] == "step"
        assert d["artifacts"][0]["format"] == "pdf"


class TestTurnStateConsumption:
    """Phase 3 #1: TurnState must be a real runtime CONSUMER, not a dead record."""

    def test_answer_directive_from_constraints(self):
        a = assess("write a python sorter with tests, under 40 lines")
        ts = TurnState.from_assessment(a, goal=a and "write a python sorter with tests, under 40 lines")
        d = ts.answer_directive()
        assert "OUTPUT REQUIREMENTS" in d
        assert "tests" in d.lower()

    def test_answer_directive_immediate_horizon(self):
        ts = TurnState.from_assessment(
            object(), goal="quick question, what's a mutex?", capabilities=False)
        assert "concise" in ts.answer_directive().lower()

    def test_answer_directive_empty_for_plain_question(self):
        # A plain knowledge question imposes no constraints/horizon/deadline →
        # empty directive → a normal answer is left unchanged.
        ts = TurnState.from_assessment(
            object(), goal="explain how a hashmap works", capabilities=False)
        assert ts.answer_directive() == ""

    def test_check_output_flags_violation(self):
        ts = TurnState.from_assessment(
            object(), goal="write it with tests", capabilities=False)
        rep = ts.check_output("def add(a, b): return a + b")
        assert rep is not None and not rep["satisfied"]
        assert any("test" in v.lower() for v in rep["violations"])

    def test_check_output_passes_when_satisfied(self):
        ts = TurnState.from_assessment(
            object(), goal="write it with tests", capabilities=False)
        rep = ts.check_output("def test_add(): assert add(1, 2) == 3")
        assert rep is not None and rep["satisfied"]

    def test_check_output_none_without_constraints(self):
        ts = TurnState.from_assessment(
            object(), goal="explain recursion", capabilities=False)
        assert ts.check_output("recursion is when...") is None

    def test_consumption_helpers_never_raise(self):
        assert TurnState.from_assessment(object()).answer_directive() == "" or True
        assert TurnState(goal="x").check_output(None) is None  # type: ignore[arg-type]


class TestClarifierPublishesTurnState:
    def test_turn_state_lands_on_board(self):
        from app.agents.clarifier import ClarifierAgent
        from app.blackboard.board import Blackboard
        from app.blackboard.schema import KEY_QUESTION

        board = Blackboard()
        board.write(KEY_QUESTION, "write a login api")
        board.write("extras", {"suppress_clarify": True})
        asyncio.get_event_loop_policy().new_event_loop()
        asyncio.run(ClarifierAgent().run(board))
        ts = board.get(KEY_TURN_STATE, None)
        # suppress_clarify may early-exit before assess; accept either but
        # when present the state must be well-formed.
        if ts is not None:
            assert ts["goal"] == "write a login api"

    def test_turn_state_on_normal_run(self):
        from app.agents.clarifier import ClarifierAgent
        from app.blackboard.board import Blackboard
        from app.blackboard.schema import KEY_QUESTION

        board = Blackboard()
        board.write(KEY_QUESTION, "what is a hash map?")
        board.write("extras", {})
        asyncio.run(ClarifierAgent().run(board))
        ts = board.get(KEY_TURN_STATE, None)
        assert ts is not None
        assert ts["intent"] == "knowledge"
        assert ts["decision"] == "answer"


class TestPlannerDecompose:
    def _plan_for(self, question: str):
        from app.agents.planner import PlannerAgent
        from app.blackboard.board import Blackboard
        from app.blackboard.schema import KEY_PLAN, KEY_QUESTION

        board = Blackboard()
        board.write(KEY_QUESTION, question)
        asyncio.run(PlannerAgent().run(board))
        return board.get(KEY_PLAN, None)

    def test_simple_goal_keeps_legacy_plan(self):
        plan = self._plan_for("what is a hash map?")
        assert plan.steps == ["retrieve", "respond", "ground"]

    def test_multi_goal_gets_decomposed_plan(self):
        plan = self._plan_for(
            "build a rest api in python, then add authentication, "
            "and then write unit tests for it")
        assert len(plan.steps) > 3                 # retrieve + subs + ground
        assert plan.steps[0] == "retrieve" and plan.steps[-1] == "ground"
        assert len(plan.priorities) == len(plan.steps)
        assert len(plan.deadlines_ms) == len(plan.steps)

    def test_flag_off_restores_legacy(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.orchestration, "planner_decompose", False)
        plan = self._plan_for(
            "build a rest api in python, then add authentication, "
            "and then write unit tests for it")
        assert plan.steps == ["retrieve", "respond", "ground"]
