"""Tests for the Failure Knowledge Base (roadmap Phase 7 #3) and its wiring into
the recovery planner (learned recovery) and the LLM engine (occurrence recording).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from app.obs import failure_kb as kb
from app.obs import plan_recovery


@pytest.fixture(autouse=True)
def _clean():
    kb.reset()
    yield
    kb.reset()


def test_occurrence_and_known():
    assert not kb.known("provider_rate_limit")
    kb.record_occurrence("provider_rate_limit")
    assert kb.known("provider_rate_limit")
    assert kb.stats()["provider_rate_limit"]["occurrences"] == 1


def test_best_recovery_needs_history():
    assert kb.best_recovery("generation_timeout") is None  # no history
    kb.record_outcome("generation_timeout", "retry_different", True)
    assert kb.best_recovery("generation_timeout") is None  # 1 attempt < min 2


def test_best_recovery_picks_winner():
    # 'retry_different' works, 'repair' doesn't — KB should prefer the winner.
    for _ in range(5):
        kb.record_outcome("generation_timeout", "retry_different", True)
        kb.record_outcome("generation_timeout", "repair", False)
    assert kb.best_recovery("generation_timeout") == "retry_different"


def test_recovery_planner_consults_kb():
    # Teach the KB a winning recovery, then confirm plan_recovery surfaces it.
    for _ in range(4):
        kb.record_outcome("provider_transport", "retry_different", True)
    plan = plan_recovery("provider_transport", attempt=1)
    assert plan.learned_action == "retry_different"


def test_recovery_planner_without_kb_history():
    plan = plan_recovery("provider_rate_limit", attempt=1)
    assert plan.learned_action is None  # nothing learned yet
    assert plan.action  # still returns the taxonomy default


def test_kb_is_fail_open():
    kb.record_outcome(None, None, True)  # type: ignore[arg-type]
    kb.record_occurrence(None)  # type: ignore[arg-type]


def _engine_src() -> str:
    return (pathlib.Path(__file__).resolve().parents[1] / "app" / "llm"
            / "engine.py").read_text(encoding="utf-8")


def test_engine_records_occurrences():
    src = _engine_src()
    tree = ast.parse(src)
    imports_kb = any(
        isinstance(n, ast.ImportFrom) and n.module == "app.obs"
        and any(a.name == "failure_kb" for a in n.names)
        for n in ast.walk(tree)
    )
    assert imports_kb, "engine must import app.obs.failure_kb"
    assert "_failure_kb.record_occurrence(_fc.id)" in src


def test_engine_records_outcomes_too():
    """The other half of the loop: without record_outcome in a production path the
    KB only ever counts occurrences and `best_recovery` stays None forever.
    (Behavioural proof lives in tests/test_recovery_engine_loop.py.)"""
    assert "_failure_kb.record_outcome(" in _engine_src(), (
        "engine must report recovery OUTCOMES to the failure KB, not just "
        "occurrences — otherwise the KB can never learn."
    )


def test_learned_action_is_what_gets_recorded():
    """The plan executes `effective_action` (the KB's pick when it has one), so the
    outcome must be attributed to THAT action, not the taxonomy default."""
    for _ in range(3):
        kb.record_outcome("provider_transport", "cooldown_wait", True)
    plan = plan_recovery("provider_transport", attempt=1)
    assert plan.action == "retry_different"          # taxonomy default
    assert plan.learned_action == "cooldown_wait"    # history says otherwise
    assert plan.effective_action == "cooldown_wait"  # …and that's what we take
