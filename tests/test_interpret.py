"""Stage-7 §3.13 — chat interpretation layer: canonicalize + structured brief."""
from __future__ import annotations

import asyncio
import types

import pytest

from app.chat import interpret as I


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.chat, "interpretation", True, raising=False)


def _stub(obj):
    async def fake(schema, msgs, **kw):
        return types.SimpleNamespace(obj=obj)
    return fake


class TestCanonicalize:
    def test_expands_safe_shorthand(self):
        assert I.canonicalize("pls fix ur code w/ tests") \
            == "please fix your code with tests"

    def test_collapses_whitespace(self):
        assert I.canonicalize("  hello   world  ") == "hello world"

    def test_strips_leading_filler(self):
        assert I.canonicalize("so, explain kafka") == "explain kafka"
        assert I.canonicalize("ok so do it") == "do it"

    def test_clean_sentence_unchanged(self):
        assert I.canonicalize("Explain how Kafka works.") \
            == "Explain how Kafka works."

    def test_does_not_corrupt_real_words(self):
        # "u" inside "ubuntu" must survive; only the whole word "u" expands.
        assert "ubuntu" in I.canonicalize("install ubuntu now")

    def test_preserves_capitalization(self):
        assert I.canonicalize("Pls help").startswith("Please")

    def test_empty_and_none(self):
        assert I.canonicalize("") == ""
        assert I.canonicalize(None) == ""          # type: ignore[arg-type]


class TestBuildBrief:
    def test_parses_a_full_brief(self, _on):
        b = _run(I.build_brief("make me a pdf report on kafka", structured_fn=_stub({
            "goal": "produce a report on Kafka",
            "deliverable_class": "artifact_only",
            "constraints": ["PDF"], "tone": "formal",
            "referents": [], "missing_slots": [], "contradictions": []})))
        assert b.goal == "produce a report on Kafka"
        assert b.deliverable_class == "artifact_only"
        assert b.constraints == ["PDF"] and b.tone == "formal"

    def test_missing_slots_drive_clarification(self, _on):
        b = _run(I.build_brief("fix it", structured_fn=_stub({
            "goal": "fix the bug", "missing_slots": ["which file"]})))
        assert b.missing_slots == ["which file"]
        assert b.needs_clarification is True

    def test_contradictions_populate(self, _on):
        b = _run(I.build_brief("make it short but very detailed",
                               structured_fn=_stub({
                                   "goal": "x",
                                   "contradictions": ["short vs very detailed"]})))
        assert b.contradictions and b.needs_clarification is True

    def test_defaults_deliverable_class_to_chat(self, _on):
        b = _run(I.build_brief("what is kafka", structured_fn=_stub({"goal": "x"})))
        assert b.deliverable_class == "chat"

    def test_fail_open_minimal_brief_on_error(self, _on):
        async def boom(schema, msgs, **kw):
            raise RuntimeError("structured down")
        b = _run(I.build_brief("do the thing", structured_fn=boom))
        assert b.goal == "do the thing" and b.missing_slots == []

    def test_invalid_obj_falls_back(self, _on):
        b = _run(I.build_brief("hello there", structured_fn=_stub(None)))
        assert b.goal == "hello there"

    def test_disabled_returns_minimal_without_calling_llm(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.chat, "interpretation", False, raising=False)
        called = {"n": 0}

        async def counted(schema, msgs, **kw):
            called["n"] += 1
            return types.SimpleNamespace(obj={"goal": "x"})
        b = _run(I.build_brief("pls fix ur code", structured_fn=counted))
        # Minimal brief = the CANONICAL text, and the LLM was never called.
        assert b.goal == "please fix your code" and called["n"] == 0

    def test_canonicalizes_before_briefing(self, _on):
        seen = {}

        async def capture(schema, msgs, **kw):
            seen["turn"] = msgs[-1]["content"]
            return types.SimpleNamespace(obj={"goal": "x"})
        _run(I.build_brief("pls do it w/ care", structured_fn=capture))
        assert "please do it with care" in seen["turn"]


class TestSchema:
    def test_schema_requires_goal(self):
        assert I.BRIEF_SCHEMA["required"] == ["goal"]

    def test_as_dict_shape(self):
        d = I.StructuredBrief(goal="g").as_dict()
        assert set(d) == {"goal", "deliverable_class", "constraints", "tone",
                          "referents", "missing_slots", "contradictions"}
