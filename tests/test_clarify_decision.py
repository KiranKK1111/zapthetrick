"""Tests for the Supervisor's clarify decision + payload helpers
(app/agents/supervisor._clarify_decision / _clarify_payload).

These are pure functions that map a parsed clarify_meta + question list onto
the answer / block / refine behavior and the SSE payload shape. They encode
the Confidence-Band, blocking, and Live-downgrade rules.
"""
from __future__ import annotations

import pytest

_sup = pytest.importorskip("app.agents.supervisor")
_decide = _sup._clarify_decision
_payload = _sup._clarify_payload


def test_answer_when_no_questions_and_no_assumptions():
    assert _decide({"confidence": 0.95}, [], "chat") == "answer"
    assert _decide({}, [], "chat") == "answer"


def test_block_when_blocking_true_with_questions():
    meta = {"blocking": True, "confidence": 0.3}
    assert _decide(meta, [{"id": "q1"}], "chat") == "block"


def test_refine_when_blocking_false_with_questions():
    meta = {"blocking": False, "confidence": 0.5}
    assert _decide(meta, [{"id": "q1"}], "chat") == "refine"


def test_assume_mode_with_assumptions_is_content_even_without_questions():
    meta = {"mode": "assume", "assumptions": [{"label": "x", "value": "y"}],
            "blocking": True}
    assert _decide(meta, [], "chat") == "block"


def test_assume_mode_without_assumptions_answers():
    meta = {"mode": "assume", "assumptions": []}
    assert _decide(meta, [], "chat") == "answer"


def test_live_downgrades_blocking_to_refine():
    meta = {"blocking": True, "confidence": 0.3}
    assert _decide(meta, [{"id": "q1"}], "live") == "refine"


def test_decision_is_deterministic():
    meta = {"blocking": False, "confidence": 0.5}
    qs = [{"id": "q1"}]
    first = _decide(meta, qs, "chat")
    for _ in range(5):
        assert _decide(meta, qs, "chat") == first


def test_payload_fills_defaults_for_missing_meta():
    p = _payload([{"id": "q1"}], None)
    assert p["questions"] == [{"id": "q1"}]
    assert p["confidence"] == 1.0
    assert p["blocking"] is False
    assert p["reason"] == ""
    assert p["estimated_questions_saved"] == 0
    assert p["mode"] == "ask"
    assert p["assumptions"] == []


def test_payload_passes_through_meta():
    meta = {"confidence": 0.4, "blocking": True, "reason": "why",
            "estimated_questions_saved": 3, "mode": "assume",
            "assumptions": [{"id": "a1", "label": "L", "value": "V"}]}
    p = _payload([], meta)
    assert p["confidence"] == 0.4 and p["blocking"] is True
    assert p["reason"] == "why" and p["estimated_questions_saved"] == 3
    assert p["mode"] == "assume" and p["assumptions"][0]["label"] == "L"
    assert p["questions"] == []
