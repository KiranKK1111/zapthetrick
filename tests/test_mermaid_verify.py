"""Mermaid verify-before-render: validate → sandbox → repair → re-verify.

The behaviour worth pinning is what happens on FAILURE. A verifier that quietly
returns "ok" when it could not run, or that replaces an unfixable diagram with an
invented one, is worse than no verifier: the user cannot tell they were lied to.
"""
from __future__ import annotations

import asyncio

import pytest

from app.diagrams import verify as V

GOOD = """graph TD
  A[Start] --> B{Decision}
  B -->|yes| C[Do the thing]
  B -->|no| D[Stop]
"""

GOOD_SUBGRAPH = """flowchart LR
  subgraph API
    A[Gateway] --> B[Service]
  end
  B --> C[(Database)]
"""


def _run(coro):
    return asyncio.run(coro)


# ── The sandbox checker ─────────────────────────────────────────────────────

def test_a_valid_diagram_passes():
    ok, errs, _warns, available = V.sandbox_check(GOOD)
    assert available, "the sandbox did not run — this test proves nothing"
    assert ok and not errs


def test_a_valid_subgraph_diagram_passes():
    ok, errs, _w, available = V.sandbox_check(GOOD_SUBGRAPH)
    assert available and ok, errs


@pytest.mark.parametrize("src,fragment", [
    ("graph TD\n  subgraph S\n  A --> B", "never closed"),
    ("graph TD\n  A --> B\n  end", "no matching"),
    ("graph TD\n  A[Fetch (REST)] --> B", "unclosed"),
    ('graph TD\n  A["unterminated --> B', "unterminated double quote"),
    ("graph TD\n  A -> B", "at least two dashes"),
    ("A --> B", "does not declare a diagram type"),
    ("graph SIDEWAYS\n  A --> B", "invalid direction"),
    ("graph TD\n  A]] --> B", "unbalanced bracket"),
])
def test_malformed_diagrams_are_caught(src, fragment):
    ok, errs, _w, available = V.sandbox_check(src)
    assert available
    assert not ok, f"should have failed: {src!r}"
    assert any(fragment in e for e in errs), f"expected {fragment!r} in {errs}"


def test_quoted_parentheses_are_accepted():
    """Wrapping a label in quotes is the CORRECT fix — flagging it would send
    the repair loop in circles."""
    ok, errs, _w, _a = V.sandbox_check('graph TD\n  A["Fetch (REST)"] --> B')
    assert ok, errs


def test_comments_and_blank_lines_are_ignored():
    src = "%% a comment\n\ngraph TD\n\n  %% another\n  A --> B\n"
    ok, errs, _w, _a = V.sandbox_check(src)
    assert ok, errs


def test_a_multi_line_diagram_with_quotes_survives_the_stdin_round_trip():
    """The source travels on stdin precisely so quotes, backticks and shell
    metacharacters cannot change the meaning of the check."""
    src = 'graph TD\n  A["say \\"hi\\" & `run`; echo $HOME"] --> B\n'
    ok, _errs, _w, available = V.sandbox_check(src)
    assert available and ok


def test_nested_subgraphs_balance_correctly():
    src = ("flowchart TB\n  subgraph outer\n    subgraph inner\n"
           "      A --> B\n    end\n  end\n")
    ok, errs, _w, _a = V.sandbox_check(src)
    assert ok, errs


# ── The full pipeline ───────────────────────────────────────────────────────

def test_a_good_diagram_needs_no_repair():
    r = _run(V.verify(GOOD))
    assert r.ok and r.repairs == 0
    assert r.source.strip() == GOOD.strip()
    assert "sandbox" in r.stages, "the sandbox stage did not run"


def test_an_empty_diagram_is_rejected_immediately():
    r = _run(V.verify("   "))
    assert not r.ok and r.errors and r.repairs == 0


def test_repair_is_attempted_and_the_result_re_verified(monkeypatch):
    calls = {"n": 0}

    async def fake_repair(source, error):
        calls["n"] += 1
        return "graph TD\n  A --> B"      # a valid diagram

    monkeypatch.setattr(V, "_repair", fake_repair)
    r = _run(V.verify("graph TD\n  A -> B"))
    assert calls["n"] == 1
    assert r.ok and r.repairs == 1
    assert "-->" in r.source


def test_an_unfixable_diagram_is_returned_UNCHANGED_with_its_errors(monkeypatch):
    """A wrong diagram is worse than a broken one: the user cannot tell it is
    wrong. So a failed repair must not invent something plausible."""
    broken = "graph TD\n  subgraph S\n  A --> B"

    seq = {"n": 0}

    async def useless_repair(source, error):
        # A DIFFERENT still-broken diagram each time. Returning the SAME string
        # twice ends the loop early by design (see the no-op test); here the
        # retry BUDGET is what should stop it.
        seq["n"] += 1
        return f"graph TD\n  subgraph S{seq['n']}\n  C{seq['n']} --> D"

    monkeypatch.setattr(V, "_repair", useless_repair)
    r = _run(V.verify(broken, max_repairs=2))
    assert not r.ok
    assert r.errors
    assert r.repairs == 2


def test_a_repair_that_changes_nothing_stops_the_loop(monkeypatch):
    """Burning the full retry budget on a model that returns its input is pure
    latency the user waits on."""
    calls = {"n": 0}

    async def no_op_repair(source, error):
        calls["n"] += 1
        return source

    monkeypatch.setattr(V, "_repair", no_op_repair)
    r = _run(V.verify("graph TD\n  A -> B", max_repairs=3))
    assert calls["n"] == 1, "should stop after the first no-op repair"
    assert not r.ok


def test_repair_can_be_disabled_for_a_report_only_pass(monkeypatch):
    async def boom(source, error):
        raise AssertionError("repair must not run when repair=False")

    monkeypatch.setattr(V, "_repair", boom)
    r = _run(V.verify("graph TD\n  A -> B", repair=False))
    assert not r.ok and r.repairs == 0


def test_an_unavailable_sandbox_is_REPORTED_not_counted_as_a_pass(monkeypatch):
    """"Could not check" and "checked and fine" are different answers. Conflating
    them is how a verifier starts lying."""
    monkeypatch.setattr(V, "sandbox_check",
                        lambda src: (True, [], [], False))
    r = _run(V.verify(GOOD, repair=False))
    assert r.sandbox_available is False
    assert "sandbox" not in r.stages


def test_a_sandbox_failure_never_blocks_a_diagram(monkeypatch):
    """Fail-open: a wedged sandbox must degrade to static validation, not stop
    the user seeing their diagram."""
    def explode(src):
        raise RuntimeError("sandbox is on fire")

    monkeypatch.setattr("app.sandbox.executor.run_code",
                        lambda *a, **k: (_ for _ in ()).throw(explode))
    ok, errs, warns, available = V.sandbox_check(GOOD)
    assert ok is True and available is False and not errs


def test_report_serializes_for_the_wire():
    r = _run(V.verify(GOOD))
    d = r.to_dict()
    assert set(d) == {"ok", "errors", "warnings", "repairs", "stages",
                      "sandbox_available"}


# ── The endpoint ────────────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch):
    """A TestClient with auth neutralized — the ROUTE is under test here, not
    authentication (which has its own suite)."""
    from fastapi.testclient import TestClient

    import app.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "auth_enforced", lambda: False)
    from app.main import app
    return TestClient(app)


def test_verify_endpoint_returns_the_report(client):
    res = client.post("/api/mermaid/verify",
                      json={"source": GOOD, "repair": False})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["source"].strip() == GOOD.strip()


def test_verify_endpoint_reports_errors_without_repairing(client):
    res = client.post("/api/mermaid/verify",
                      json={"source": "graph TD\n  A -> B", "repair": False})
    body = res.json()
    assert body["ok"] is False
    assert body["errors"]
    assert body["repairs"] == 0
