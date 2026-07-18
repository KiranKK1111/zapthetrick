"""Unified response envelope — response.v1 (Architecture.md §5)."""
from __future__ import annotations

from app.response_arch.envelope import (
    ResponseEnvelope, build_envelope, structure_suggestions)


def test_schema_version_and_defaults():
    env = build_envelope()
    assert env["schema"] == "response.v1"
    assert env["answer"] == {"incomplete": False}
    assert env["topic_shift"] is False
    # absent optional fields are omitted
    assert "document" not in env
    assert "grounding" not in env
    assert "intent" not in env


def test_full_envelope_roundtrip():
    env = build_envelope(
        conversation_id="c1", message_id="m1",
        intent={"type": "code_generation", "confidence": 0.82, "source": "semantic"},
        difficulty="hard",
        resolved_prompt="reverse a string in python",
        topic_shift=False,
        answer_shape="code", incomplete=False,
        suggestions=[{"text": "Add unit tests", "intent_hint": "test_generation"}],
        document={"generate": True, "format": "py", "formats": ["py"]},
        artifacts=[{"kind": "code", "filename": "sort.py"}],
        grounding={"unsupported": []},
        model="some-model", latency_ms=1234, degraded=[],
        confidence_band="high", route={"category": "coding", "strategy": "single"},
    )
    assert env["schema"] == "response.v1"
    assert env["intent"]["type"] == "code_generation"
    assert env["difficulty"] == "hard"
    assert env["answer"] == {"shape": "code", "incomplete": False}
    assert env["suggestions"][0]["intent_hint"] == "test_generation"
    assert env["document"]["format"] == "py"
    assert env["meta"]["model"] == "some-model"
    assert env["meta"]["latency_ms"] == 1234
    assert env["meta"]["route"]["category"] == "coding"


def test_incomplete_and_empty_meta_omitted():
    env = build_envelope(incomplete=True, model=None, latency_ms=None)
    assert env["answer"]["incomplete"] is True
    # meta had only None/empty values → omitted entirely
    assert "meta" not in env or env["meta"] == {}


def test_validates_as_model():
    env = build_envelope(intent={"type": "knowledge"})
    # Re-parse to confirm it's a valid response.v1 object.
    parsed = ResponseEnvelope.model_validate({**env, "schema": env["schema"]})
    assert parsed.schema_ == "response.v1"
    assert parsed.intent == {"type": "knowledge"}


# --- load path: persisted envelope + legacy reconstruction -----------------

def _row(**kw):
    from types import SimpleNamespace
    base = dict(role="assistant", id="m1", envelope=None, sources=None,
                intent=None, incomplete=False, model=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_load_returns_persisted_envelope_with_message_id():
    from app.api.routes_chat import _envelope_for
    stored = {"schema": "response.v1", "intent": {"type": "code_generation"},
              "answer": {"incomplete": False}, "topic_shift": False}
    out = _envelope_for(_row(id="abc", envelope=stored))
    assert out["intent"]["type"] == "code_generation"
    assert out["message_id"] == "abc"          # stamped from the row id


def test_load_reconstructs_legacy_row_without_envelope():
    from app.api.routes_chat import _envelope_for
    out = _envelope_for(_row(
        id="xyz", envelope=None, intent="knowledge", incomplete=True,
        model="some-model",
        sources={"document": True, "format": "pdf", "formats": ["pdf"]},
    ))
    assert out["schema"] == "response.v1"
    assert out["intent"]["type"] == "knowledge"
    assert out["answer"]["incomplete"] is True
    assert out["document"]["format"] == "pdf"
    assert out["meta"]["model"] == "some-model"


def test_load_user_message_has_no_envelope():
    from app.api.routes_chat import _envelope_for
    assert _envelope_for(_row(role="user")) is None


# --- first-class suggestions (Architecture §6) ------------------------------

def test_structure_suggestions_from_strings():
    out = structure_suggestions(["Add unit tests", "  ", "Handle edge cases"])
    assert out == [
        {"text": "Add unit tests", "source": "profile"},
        {"text": "Handle edge cases", "source": "profile"},
    ]  # blank dropped


def test_structure_suggestions_with_intent_hint():
    out = structure_suggestions(
        ["write tests for this"], intent_of=lambda t: "test_generation")
    assert out[0]["intent_hint"] == "test_generation"


def test_structure_suggestions_passthrough_dicts_and_source():
    out = structure_suggestions(
        [{"text": "Related: refresh tokens", "source": "knowledge_graph",
          "intent_hint": "knowledge"},
         "continue the dashboard"],
        source="memory_graph")
    assert out[0] == {"text": "Related: refresh tokens",
                      "source": "knowledge_graph", "intent_hint": "knowledge"}
    assert out[1] == {"text": "continue the dashboard", "source": "memory_graph"}


def test_envelope_carries_suggestions():
    env = build_envelope(suggestions=structure_suggestions(["Add tests"]))
    assert env["suggestions"] == [{"text": "Add tests", "source": "profile"}]


def test_envelope_carries_trace_and_trace_id():
    env = build_envelope(trace={"id": "t9", "model": "m", "tools": ["retriever"]})
    assert env["meta"]["trace_id"] == "t9"
    assert env["meta"]["trace"]["tools"] == ["retriever"]
