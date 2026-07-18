"""Constrained decoding wiring into the generation path (P6 #24)."""
from __future__ import annotations

import asyncio

import app.response_arch.structured as S

SCHEMA = {
    "type": "object",
    "required": ["title", "score"],
    "properties": {
        "title": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
    },
}


def test_structured_options_gated_on_capability():
    assert S.structured_options(SCHEMA, model_meta={"supports_json": True})
    assert S.structured_options(SCHEMA, model_meta={"supports_json": True})[
        "response_format"]["type"] == "json_schema"
    # incapable model → no response_format (validate-and-repair still applies)
    assert S.structured_options(SCHEMA, model_meta={}) == {}
    assert S.structured_options(SCHEMA, model_meta=None) == {}


def test_enforce_valid_and_invalid():
    obj, errs = S.enforce('{"title":"x","score":5}', SCHEMA)
    assert obj and errs == []
    _, errs2 = S.enforce('{"score":99}', SCHEMA)
    assert errs2  # missing title + out of bounds


def test_repair_prompt_mentions_schema_and_errors():
    p = S.repair_prompt(SCHEMA, '{"score":99}', ["$.title: required"])
    assert "schema" in p.lower() and "title" in p


class _FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    async def complete_routed(self, messages, options=None):
        self.calls += 1
        return self._replies.pop(0), "fake-model"


def test_generate_structured_happy_path(monkeypatch):
    fake = _FakeLLM(['{"title":"ok","score":3}'])
    import app.core.llm_client as lc
    monkeypatch.setattr(lc, "llm", fake, raising=False)
    obj, errs = asyncio.run(
        S.generate_structured([{"role": "user", "content": "go"}], SCHEMA))
    assert obj["title"] == "ok" and errs == []
    assert fake.calls == 1


def test_generate_structured_repairs_once(monkeypatch):
    fake = _FakeLLM(['{"score":"bad"}', '{"title":"fixed","score":7}'])
    import app.core.llm_client as lc
    monkeypatch.setattr(lc, "llm", fake, raising=False)
    obj, errs = asyncio.run(
        S.generate_structured([{"role": "user", "content": "go"}], SCHEMA))
    assert obj["title"] == "fixed" and errs == []
    assert fake.calls == 2                 # one repair round


def test_generate_structured_fail_open(monkeypatch):
    class _Boom:
        async def complete_routed(self, *a, **k):
            raise RuntimeError("provider down")

    import app.core.llm_client as lc
    monkeypatch.setattr(lc, "llm", _Boom(), raising=False)
    obj, errs = asyncio.run(
        S.generate_structured([{"role": "user", "content": "go"}], SCHEMA))
    assert obj is None and errs


def test_wiring_note_is_actionable():
    assert "generate_structured" in S.wiring_note()
