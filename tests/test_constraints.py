"""Tests for the Constraint Solver (roadmap Phase 4 #7), incl. wiring into TurnState."""
from __future__ import annotations

from app.core import constraints as C
from app.core.world_state import TurnState


# ── extraction ──────────────────────────────────────────────────────────────
def test_extract_include_and_limit():
    cs = C.extract_constraints("write a python sorter with tests, under 50 lines")
    kinds = {(c.kind, c.key) for c in cs}
    assert (C.MUST_INCLUDE, "tests") in kinds
    assert (C.MAX_LINES, "50") in kinds


def test_extract_avoid_and_format():
    cs = C.extract_constraints("return the data as JSON, no external dependencies")
    kinds = {(c.kind, c.key) for c in cs}
    assert (C.FORMAT, "json") in kinds
    assert (C.MUST_AVOID, "external dependencies") in kinds


def test_extract_nothing_from_plain():
    assert C.extract_constraints("explain how a hashmap works") == []


# ── checking ────────────────────────────────────────────────────────────────
def test_check_must_include_pass_and_fail():
    c = [C.Constraint(C.MUST_INCLUDE, "tests", "must include tests")]
    ok = C.check("def test_foo(): assert add(1,2)==3", c)
    assert ok.satisfied and ok.checked == 1
    bad = C.check("def add(a,b): return a+b", c)
    assert not bad.satisfied and "must include tests" in bad.violations


def test_check_max_lines():
    c = [C.Constraint(C.MAX_LINES, "3", "at most 3 lines")]
    assert C.check("a\nb\nc", c).satisfied
    assert not C.check("a\nb\nc\nd\ne", c).satisfied


def test_check_json_format():
    c = [C.Constraint(C.FORMAT, "json", "must be valid JSON")]
    assert C.check('{"a": 1}', c).satisfied
    assert not C.check("not json", c).satisfied


def test_unverifiable_constraint_is_unchecked_not_violated():
    # 'no recursion' can't be verified by absence → unchecked, never a violation.
    c = [C.Constraint(C.MUST_AVOID, "recursion", "must avoid recursion")]
    rep = C.check("def f(n): return f(n-1)", c)
    assert rep.satisfied  # not flagged
    assert "must avoid recursion" in rep.unchecked


def test_check_is_fail_open():
    assert C.check(None, None).satisfied  # type: ignore[arg-type]


# ── wiring ──────────────────────────────────────────────────────────────────
def test_wired_into_turnstate():
    class _A:
        intent = "coding"; decision = "answer"; confidence = 0.8; ambiguity = 0.1
        risk = 0.0; risk_level = "low"; missing_required = []; matrix = None; policy = None
    ts = TurnState.from_assessment(_A(), goal="write a parser with tests under 100 lines",
                                   capabilities=False)
    keys = {(c["kind"], c["key"]) for c in ts.constraints}
    assert (C.MUST_INCLUDE, "tests") in keys and (C.MAX_LINES, "100") in keys
    assert "constraints" in ts.as_dict()
