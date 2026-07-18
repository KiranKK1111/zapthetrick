"""Tool / structured-output capability routing (intelligent-model-routing R4,
task 5.2). Pins Property 4 on the pure filter: restriction to capable models,
full-pool fallback + unmet-constraint record when none qualify, and that the
filter never touches vision routing.
"""
from __future__ import annotations

from app.llm.router import apply_capability_filter


def _pool():
    return [
        {"model": "a", "supports_tools": True, "supports_json": True, "score": 1},
        {"model": "b", "supports_tools": False, "supports_json": True, "score": 2},
        {"model": "c", "supports_tools": True, "supports_json": False, "score": 3},
    ]


def test_tool_filter_restricts_to_capable():
    out = apply_capability_filter(_pool(), needs_tool=True, needs_json=False)
    assert {c["model"] for c in out} == {"a", "c"}


def test_json_filter_restricts_to_capable():
    out = apply_capability_filter(_pool(), needs_tool=False, needs_json=True)
    assert {c["model"] for c in out} == {"a", "b"}


def test_combined_tool_and_json():
    out = apply_capability_filter(_pool(), needs_tool=True, needs_json=True)
    assert {c["model"] for c in out} == {"a"}


def test_no_capable_model_falls_back_and_records_unmet():
    pool = [{"model": "x", "supports_tools": False, "supports_json": False,
             "score": 1}]
    trace: list = []
    out = apply_capability_filter(pool, needs_tool=True, needs_json=False, trace=trace)
    assert out == pool                              # full-pool fallback (R4.3)
    assert any(t.get("unmet_constraint") == "tool" for t in trace)


def test_no_constraints_is_identity():
    pool = _pool()
    assert apply_capability_filter(pool, needs_tool=False, needs_json=False) is pool
