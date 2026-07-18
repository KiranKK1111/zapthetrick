"""Routing explainability (intelligent-model-routing R10, task 13.4 backend).

Pins Property 10: the capability filter records an unmet constraint into the
trace, and the meta-router decision carries an inspectable trace of the chosen
strategy/category/difficulty.
"""
from __future__ import annotations

from app.llm.router import apply_capability_filter
from app.llm import meta_router as M


def test_capability_filter_records_unmet_constraint_in_trace():
    pool = [{"model": "x", "supports_tools": False, "supports_json": False,
             "score": 1}]
    trace: list = []
    apply_capability_filter(pool, needs_tool=True, needs_json=True, trace=trace)
    scopes = {t.get("unmet_constraint") for t in trace}
    assert "tool" in scopes and "json" in scopes


def test_meta_router_decision_has_trace():
    dec = M.decide({"text": "write a python function", "difficulty": "hard"},
                   enabled=True)
    assert dec.trace and dec.trace[0]["category"] == "coding"
    assert dec.trace[0]["difficulty"] == "hard"
    assert "strategy" in dec.trace[0]


def test_route_kwargs_carry_capability_requirements():
    dec = M.decide({"text": "return strict json", "difficulty": "standard",
                    "needs_json": True, "needs_tool": True}, enabled=True)
    kw = dec.route_kwargs()
    assert kw["needs_json"] is True and kw["needs_tool"] is True
