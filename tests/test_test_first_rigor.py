"""P2-5 — test-first rigor.

Pure/offline: the test-file heuristic, untested-symbol detection, the test
surface over a real git workspace, the `test_plan` tool, the confidence signals
for test coverage, and the run_goal strict test gate (LLM scripted).
"""
from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from app.agent.testgen import (
    TEST_FIRST_DIRECTIVE,
    is_test_file,
    leaf_name,
    untested_symbols,
)
from app.agent.testgen import TestSurface as Surface
from app.agent.testgen import test_surface as compute_surface
from app.chat.trust import ConfidenceSignals, confidence_band


# ── pure helpers ──────────────────────────────────────────────────────────
def test_is_test_file():
    assert is_test_file("tests/test_foo.py")
    assert is_test_file("src/foo_test.go")
    assert is_test_file("app/bar.test.ts")
    assert is_test_file("spec/user_spec.rb")
    assert is_test_file("com/example/UserTest.java")
    assert not is_test_file("src/foo.py")
    assert not is_test_file("app/main.go")


def test_leaf_name():
    assert leaf_name("User.save") == "save"
    assert leaf_name("module::func") == "func"
    assert leaf_name("plain") == "plain"
    assert leaf_name("") == ""


def test_untested_symbols():
    syms = ["User.login", "User.logout", "slugify"]
    tests = ["def test_login(): assert User.login()"]
    out = untested_symbols(syms, tests)
    assert "User.logout" in out and "slugify" in out
    assert "User.login" not in out          # 'login' referenced in a test


def test_untested_symbols_no_tests_means_all_untested():
    assert untested_symbols(["a", "b"], []) == ["a", "b"]
    assert untested_symbols([], ["anything"]) == []


# ── confidence signals ──────────────────────────────────────────────────────
def test_confidence_rewards_tests_penalizes_untested():
    base = confidence_band(ConfidenceSignals(goal_passed=True,
                                             verify_attempted=True,
                                             verify_ok=True))
    with_tests = confidence_band(ConfidenceSignals(
        goal_passed=True, verify_attempted=True, verify_ok=True,
        tests_added=3))
    untested = confidence_band(ConfidenceSignals(
        goal_passed=True, verify_attempted=True, verify_ok=True,
        untested_changes=3))
    assert with_tests.score >= base.score
    assert untested.score < base.score
    assert any("without a test" in r for r in untested.reasons)


# ── test surface over a real git workspace ───────────────────────────────────
def _git(ws, *args):
    subprocess.run(["git", *args], cwd=ws, check=True,
                   capture_output=True, text=True)


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_test_surface_flags_untested_added_symbol(tmp_path):
    ws = str(tmp_path)
    # baseline repo with a test runner (pyproject → python build system)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "mod.py").write_text("def existing():\n    return 1\n",
                                     encoding="utf-8")
    _git(ws, "init")
    _git(ws, "config", "user.email", "t@t.com")
    _git(ws, "config", "user.name", "t")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-m", "base")

    # add a new symbol WITHOUT a test
    (tmp_path / "mod.py").write_text(
        "def existing():\n    return 1\n\n"
        "def brand_new_feature():\n    return 2\n", encoding="utf-8")

    surface = asyncio.run(compute_surface(ws))
    assert "brand_new_feature" in surface.added
    assert "brand_new_feature" in surface.untested
    assert surface.has_test_system is True
    assert surface.untested_count >= 1


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_test_surface_counts_added_tests_as_covered(tmp_path):
    ws = str(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "mod.py").write_text("def existing():\n    return 1\n",
                                     encoding="utf-8")
    os.makedirs(tmp_path / "tests")
    (tmp_path / "tests" / "test_mod.py").write_text(
        "def test_existing(): pass\n", encoding="utf-8")
    _git(ws, "init")
    _git(ws, "config", "user.email", "t@t.com")
    _git(ws, "config", "user.name", "t")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-m", "base")

    # add a feature AND its test
    (tmp_path / "mod.py").write_text(
        "def existing():\n    return 1\n\n"
        "def shiny():\n    return 9\n", encoding="utf-8")
    (tmp_path / "tests" / "test_mod.py").write_text(
        "def test_existing(): pass\n\ndef test_shiny(): assert shiny() == 9\n",
        encoding="utf-8")

    surface = asyncio.run(compute_surface(ws))
    assert "shiny" in surface.added
    assert surface.tests_added >= 1
    assert "shiny" not in surface.untested     # covered by test_shiny


# ── test_plan tool ────────────────────────────────────────────────────────────
def test_test_plan_tool_no_changes(tmp_path):
    from app.agent import tools
    res = asyncio.run(tools.test_plan(str(tmp_path)))
    assert "nothing staged" in res.lower() or "no code symbols" in res.lower()


# ── run_goal strict test gate (LLM scripted) ─────────────────────────────────
def test_run_goal_strict_gate_blocks_then_allows(monkeypatch, tmp_path):
    """With require_tests, a passing round that left untested symbols is sent
    back for tests; once covered (or on the final round) it completes."""
    from app.agent import loop

    ws = str(tmp_path)

    # Scripted: every run_agent step just finalizes immediately.
    async def fake_run_agent(prompt, **kwargs):
        yield {"type": "final", "message": "did it"}
    monkeypatch.setattr(loop, "run_agent", fake_run_agent)

    # No real verification.
    async def fake_verify(*a, **k):
        class _R:
            attempted = False
            ok = True
            summary = ""
            def feedback(self):
                return ""
        return _R()
    monkeypatch.setattr("app.agent_workspace.verify.verify_workspace",
                        fake_verify)

    # Evaluator always passes.
    async def fake_eval(condition, workspace):
        return True, ""
    monkeypatch.setattr(loop, "_evaluate", fake_eval)

    # First surface call → untested; after that → clean.
    calls = {"n": 0}

    async def fake_surface(workspace, **k):
        calls["n"] += 1
        s = Surface(has_test_system=True)
        if calls["n"] == 1:
            s.added = ["foo"]
            s.untested = ["foo"]
        return s
    monkeypatch.setattr("app.agent.testgen.test_surface", fake_surface)

    events = asyncio.run(_collect(loop.run_goal(
        "build it", "done?", workspace=ws, max_rounds=3, require_tests=True)))
    evals = [e for e in events if e["type"] == "goal_eval"]
    # round 1 blocked for tests, round 2 allowed
    assert any(e.get("tests_required") for e in evals)
    done = [e for e in events if e["type"] == "goal_done"]
    assert done and done[-1]["passed"] is True


async def _collect(agen):
    return [e async for e in agen]


def test_directive_is_nonempty():
    assert "TEST-FIRST" in TEST_FIRST_DIRECTIVE
    assert "characterization" in TEST_FIRST_DIRECTIVE.lower()
