"""Stage-7 §3.7 — clarification engine vNext: materiality × cost decision policy."""
from __future__ import annotations

import types

import pytest

from app.clarify import policy as P


def _brief(missing=None, contra=None):
    return types.SimpleNamespace(missing_slots=missing or [],
                                 contradictions=contra or [])


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.decision_core, "clarify_v2", True, raising=False)
    monkeypatch.setattr(cfg.decision_core, "clarify_budget", 2, raising=False)


class TestMateriality:
    def test_language_is_material(self):
        assert P.is_material("programming language") is True
        assert P.is_material("output format") is True

    def test_tone_is_not_material(self):
        assert P.is_material("tone") is False
        assert P.is_material("verbosity") is False


class TestDecide:
    def test_material_slot_is_asked(self, _on):
        d = P.decide(_brief(missing=["programming language"]))
        assert d.action == "ask" and "language" in d.slot.lower()
        assert d.question

    def test_non_material_is_assumed_not_asked(self, _on):
        d = P.decide(_brief(missing=["tone", "verbosity"]))
        assert d.action == "assume"
        assert {a.slot for a in d.assumptions} == {"tone", "verbosity"}

    def test_one_question_per_turn_rest_assumed(self, _on):
        # Two material slots → ask ONE, assume the other (+ any non-material).
        d = P.decide(_brief(missing=["language", "target platform", "tone"]))
        assert d.action == "ask"
        assumed = {a.slot for a in d.assumptions}
        assert "target platform" in assumed and "tone" in assumed

    def test_contradiction_always_asks_first(self, _on):
        d = P.decide(_brief(missing=["language"],
                            contra=["short but exhaustive"]))
        assert d.action == "ask" and d.slot == "contradiction"
        assert "conflict" in d.question.lower()

    def test_budget_exhausted_switches_to_assume(self, _on):
        d = P.decide(_brief(missing=["output format"]), asked=2)
        assert d.action == "assume"
        assert d.assumptions[0].value == "Markdown"    # a labeled default

    def test_sticky_resolved_slot_not_reasked(self, _on):
        d = P.decide(_brief(missing=["language"]), resolved={"language"})
        assert d.action == "proceed"

    def test_no_missing_proceeds(self, _on):
        assert P.decide(_brief()).action == "proceed"

    def test_disabled_always_proceeds(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.decision_core, "clarify_v2", False, raising=False)
        assert P.decide(_brief(missing=["language"],
                               contra=["a vs b"])).action == "proceed"

    def test_never_raises_on_bad_brief(self, _on):
        assert P.decide(object()).action in ("proceed", "assume", "ask")


class TestAssumeAndLabel:
    def test_defaults_are_labeled_with_a_reason(self, _on):
        d = P.decide(_brief(missing=["tone"]))
        a = d.assumptions[0]
        assert a.slot == "tone" and a.value and a.why   # value + why present


class TestAssumptionLedger:
    def test_records_and_dedups(self):
        led = P.AssumptionLedger()
        led.record([P.Assumption("language", "Python", "common"),
                    P.Assumption("tone", "neutral", "default")])
        led.record([P.Assumption("language", "Go", "changed")])   # sticky: first wins
        assert led.slots() == {"language", "tone"}
        entry = {a["slot"]: a["value"] for a in led.as_list()}
        assert entry["language"] == "Python"          # first assumption stuck

    def test_slots_feed_sticky_resolution(self, _on):
        led = P.AssumptionLedger()
        led.record(P.decide(_brief(missing=["tone"])).assumptions)
        # Next turn: the assumed slot is sticky → not re-assumed/asked.
        d = P.decide(_brief(missing=["tone"]), resolved=led.slots())
        assert d.action == "proceed"
