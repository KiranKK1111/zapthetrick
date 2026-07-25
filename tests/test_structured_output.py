"""Schema-enforced structured output (vNext §8.7).

Pins the repair→validate→retry-with-rotation→honest-degrade ladder with a
STUBBED routing engine (no model calls). Async driven via asyncio.run.
"""
from __future__ import annotations

import asyncio

from app.llm import structured as S

_SCHEMA = {
    "type": "object",
    "properties": {"lang": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["lang"],
}


class _Route:
    def __init__(self, mid=1, name="fake-model"):
        self.model_db_id = mid
        self.display_name = name
        self.model_id = name


def _stub(monkeypatch, *texts, raise_exc=None):
    """Patch engine.route_and_complete to return `texts` in order (last repeats).
    Returns a dict recording call count + the avoid_model_db_id each call saw."""
    rec = {"n": 0, "avoid": [], "preferred": []}

    async def fake(messages, options, *, session_key=None,
                   preferred_model_db_id=None):
        if raise_exc is not None:
            raise raise_exc
        i = rec["n"]
        rec["n"] += 1
        rec["avoid"].append(options.get("avoid_model_db_id"))
        rec["preferred"].append(preferred_model_db_id)
        return texts[min(i, len(texts) - 1)], _Route(mid=1 + i)

    monkeypatch.setattr("app.llm.engine.route_and_complete", fake)
    return rec


def test_valid_first_try(monkeypatch):
    _stub(monkeypatch, '{"lang": "dart", "n": 3}')
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}]))
    assert res.valid
    assert res.obj == {"lang": "dart", "n": 3}
    assert res.degraded == []


def test_fenced_and_trailing_comma_repaired(monkeypatch):
    _stub(monkeypatch, '```json\n{"lang": "go", "n": 1,}\n```')
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}]))
    assert res.valid
    assert res.obj["lang"] == "go"
    assert res.degraded == []   # deterministic repair, no retry needed


def test_retry_with_error_feedback_and_model_rotation(monkeypatch):
    # First emission is type-invalid (lang is a number); second is valid.
    rec = _stub(monkeypatch, '{"lang": 5}', '{"lang": "rust"}')
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}]))
    assert res.valid
    assert res.obj == {"lang": "rust"}
    assert "schema_retry" in res.degraded
    assert rec["n"] == 2
    # The retry must ROTATE away from the first attempt's model.
    assert rec["avoid"][1] == 1


def test_persistent_invalid_degrades_honestly(monkeypatch):
    _stub(monkeypatch, '{"lang": 5}')   # always type-invalid
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}],
                                   retries=1))
    assert res.valid is False
    assert res.errors                     # carries the validation error
    assert "schema_invalid" in res.degraded


def test_unparseable_degrades_to_unvalidated(monkeypatch):
    _stub(monkeypatch, "this is not json at all")
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}],
                                   retries=0))
    assert res.valid is False
    assert res.obj is None
    assert "schema_unvalidated" in res.degraded


def test_generation_error_is_captured_not_raised(monkeypatch):
    _stub(monkeypatch, raise_exc=RuntimeError("no route"))
    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}]))
    assert res.valid is False
    assert res.obj is None
    assert "generation_error" in res.degraded
    assert any("generation failed" in e for e in res.errors)


def test_parse_with_repair_handles_top_level_array():
    # Regression: extract_json used to be object-biased and dropped the outer
    # [ ] of a top-level array. Also exercises fence-strip + trailing-comma.
    schema = {"type": "array", "items": {
        "type": "object",
        "properties": {"input": {"type": "string"},
                       "expected": {"type": "string"}},
        "required": ["input", "expected"]}}
    txt = ('Sure:\n```json\n[{"input":"1","expected":"2"},'
           '{"input":"3","expected":"4"},]\n```')
    obj, errs = S.parse_with_repair(txt, schema)
    assert obj == [{"input": "1", "expected": "2"},
                   {"input": "3", "expected": "4"}]
    assert errs == []


def test_gbnf_covers_the_schema():
    g = S.schema_to_gbnf(_SCHEMA)
    assert "root ::=" in g
    assert "lang" in g          # property name appears in the object rule
    # enum schema → an enum rule with the literal options
    eg = S.schema_to_gbnf({"type": "string", "enum": ["a", "b"]})
    assert '"a"' in eg and '"b"' in eg
