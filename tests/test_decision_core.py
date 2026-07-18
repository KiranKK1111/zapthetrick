"""Phase-1 decision core (ArchitectureVerdict.md): requirement matrix, numeric
risk scoring, assumption persistence. Deterministic — no LLM, no DB.

Contract under test:
  * The matrix NEVER disagrees with the pre-gate's flat lists (compat).
  * Provenance: prompt-named slots attribute to "prompt", known prefs to
    "preference".
  * Risk: read-only asks are LOW (negative band delta = interrupt less);
    destructive/production work is HIGH (positive delta = clarify sooner);
    the assessor never raises (neutral fallback).
  * Assumptions: recorded → suppressed immediately (provisional), promoted on
    a non-objecting next turn, cleared on an objection.
"""
from __future__ import annotations

import pytest

from app.clarify import intent_pipeline as ip
from app.clarify.goal_ledger import GoalLedger
from app.clarify.requirement_matrix import (
    SOURCE_ATTACHMENT, SOURCE_PREFERENCE, SOURCE_PROMPT, build_matrix)
from app.clarify.risk import HIGH, LOW, MEDIUM, assess_risk


# ---------------------------------------------------------------- matrix ----
class TestRequirementMatrix:
    def test_matrix_mirrors_flat_lists_code_gen_no_language(self):
        a = ip.assess("write a binary search function")
        assert a.matrix is not None
        assert a.matrix.missing_required() == a.missing_required
        assert a.matrix.missing_optional() == a.missing_optional
        assert "language" in a.matrix.missing_required()

    def test_matrix_mirrors_flat_lists_when_complete(self):
        a = ip.assess("write a binary search function in python")
        assert a.matrix is not None
        assert a.matrix.missing_required() == a.missing_required == []
        assert a.matrix.available().get("language") == "python"

    def test_prompt_provenance(self):
        a = ip.assess("write a login api in python")
        fact = a.matrix.facts.get("language")
        assert fact is not None and fact.value == "python"
        assert fact.source == SOURCE_PROMPT
        assert fact.confidence >= 0.9

    def test_preference_provenance(self):
        # Language known from prefs, not named this turn → preference source.
        m = build_matrix("code_gen", {"language": "python"}, [], [],
                         text_slots={"language": None},
                         known_prefs={"language": "python"})
        fact = m.facts.get("language")
        assert fact is not None and fact.source == SOURCE_PREFERENCE

    def test_fill_upgrades_only_on_higher_confidence(self):
        m = build_matrix("code_gen", {}, ["language"], [], text_slots={})
        assert m.fill("language", "java", SOURCE_ATTACHMENT)
        assert m.available()["language"] == "java"
        assert "language" not in m.missing_required()
        # A weaker default must NOT displace the attachment evidence.
        assert not m.fill("language", "python", "default")
        assert m.available()["language"] == "java"

    def test_project_build_composite_slot(self):
        a = ip.assess("build a web app in django")
        assert a.matrix is not None
        assert "language_or_framework" not in a.matrix.missing_required()

    def test_as_dict_serializes(self):
        a = ip.assess("write a rest api")
        d = a.matrix.as_dict()
        assert d["intent"] == a.intent
        assert isinstance(d["facts"], list)
        assert d["missing_required"] == a.missing_required


# ------------------------------------------------------------------ risk ----
class TestRiskScoring:
    @pytest.mark.parametrize("text,intent", [
        ("what is a hash map?", "knowledge"),
        ("explain this code", "knowledge"),
        ("compare kafka and rabbitmq", "comparison"),
    ])
    def test_read_only_is_low_risk(self, text, intent):
        ra = assess_risk(text, intent)
        assert ra.level == LOW
        assert ra.band_delta < 0          # interrupt less

    def test_destructive_is_high_risk(self):
        ra = assess_risk("write a script to delete all rows from the "
                         "production database", "code_gen")
        assert ra.level == HIGH
        assert ra.band_delta > 0          # clarify sooner
        assert "destructive_operation" in ra.reasons
        assert "production_environment" in ra.reasons

    def test_infrastructure_raises_risk(self):
        ra = assess_risk("generate terraform to deploy the backend",
                         "code_gen")
        assert ra.level in (MEDIUM, HIGH)
        assert "deployment_or_infrastructure" in ra.reasons

    def test_project_build_base_is_medium(self):
        ra = assess_risk("build me a todo app", "project_build")
        assert ra.level in (MEDIUM, HIGH)

    def test_never_raises_neutral_fallback(self):
        ra = assess_risk(None, None, None)          # type: ignore[arg-type]
        assert 0.0 <= ra.score <= 1.0

    def test_assess_attaches_risk_fields(self):
        a = ip.assess("drop the users table in production")
        assert a.risk_level == HIGH
        assert a.risk_band_delta > 0
        assert any(r.startswith("high_risk:") for r in a.reasons)

    def test_risk_never_changes_the_pregate_decision(self):
        # Same prompt, risk on/off → identical decision (delta only feeds
        # the downstream band).
        a = ip.assess("write a script to wipe temp files in python")
        b_decision = a.decision
        assert b_decision in (ip.ANSWER, ip.CLARIFY, ip.DEFER)
        assert a.decision == b_decision


# ---------------------------------------------------------- assumptions ----
class TestAssumptionPersistence:
    def _ledger(self):
        return GoalLedger({}, "conv-1")

    def test_record_creates_provisional_and_suppresses(self):
        led = self._ledger()
        led.record_assumptions(["Assuming Python since none was specified"])
        assert led.assumptions("provisional")
        # Provisional slot suppresses re-asking immediately.
        assert led.confirmed_slots().get("language") == "python"

    def test_silence_promotes(self):
        led = self._ledger()
        led.record_assumptions(["Assuming Python"])
        led.observe("great, also add unit tests please")
        assert led.assumptions("accepted")
        assert not led.assumptions("provisional")
        # Now a CONFIRMED slot (survives independently of provisional store).
        assert led.confirmed_slots().get("language") == "python"

    def test_objection_clears(self):
        led = self._ledger()
        led.record_assumptions(["Assuming Python"])
        led.observe("no, use java instead")
        assert led.assumptions("rejected")
        assert not led.assumptions("provisional")
        # The objection turn itself names java → extracted as the real slot.
        assert led.confirmed_slots().get("language") == "java"

    def test_no_cid_is_noop(self):
        led = GoalLedger({}, None)
        led.record_assumptions(["Assuming Python"])
        assert led.assumptions() == []

    def test_empty_and_none_safe(self):
        led = self._ledger()
        led.record_assumptions(None)
        led.record_assumptions([])
        led.record_assumptions(["", None])          # type: ignore[list-item]
        assert led.assumptions() == []
