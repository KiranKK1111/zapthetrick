"""Constrained Decoding / Structured Outputs (P6 #24)."""
from __future__ import annotations

from app.llm import constrained as C

SCHEMA = {
    "type": "object",
    "required": ["title", "score"],
    "properties": {
        "title": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "tags": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "enum": ["a", "b"]},
    },
}


def test_extract_json_from_fenced_and_chatty():
    assert C.parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert C.parse_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_valid_object_passes():
    obj, errs = C.coerce('{"title":"x","score":7,"kind":"a"}', SCHEMA)
    assert obj is not None and errs == []


def test_missing_required_and_wrong_type():
    _, errs = C.coerce('{"score":"nope"}', SCHEMA)
    assert any("title" in e and "required" in e for e in errs)
    assert any("expected integer" in e for e in errs)


def test_bounds_and_enum():
    _, errs = C.coerce('{"title":"x","score":99,"kind":"z"}', SCHEMA)
    assert any("maximum" in e for e in errs)
    assert any("not in" in e for e in errs)


def test_array_items_validated():
    _, errs = C.coerce('{"title":"x","score":1,"tags":["ok",5]}', SCHEMA)
    assert any("[1]" in e and "expected string" in e for e in errs)


def test_unparseable_returns_none():
    obj, errs = C.coerce("not json at all", SCHEMA)
    assert obj is None and errs == ["not valid JSON"]


def test_response_format_payload_and_capability():
    rf = C.response_format(SCHEMA, name="resp")
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] is SCHEMA
    assert C.supports_structured({"supports_json": True}) is True
    assert C.supports_structured({}) is False
