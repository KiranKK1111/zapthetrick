"""Wiring test — the failure taxonomy is actually USED in a live path, not dead
code. The LLM engine's retry loop calls `failure_taxonomy.observe()` on every
ProviderError, so provider failures get classified for diagnostics (Phase 1 #5).
"""
from __future__ import annotations

import ast
import pathlib

from app.llm.providers import ProviderError
from app.obs import classify_exception, observe

_ENGINE = pathlib.Path(__file__).resolve().parents[1] / "app" / "llm" / "engine.py"


def test_engine_imports_and_calls_taxonomy():
    src = _ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports_it = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "app.obs"
        and any(a.name == "failure_taxonomy" for a in n.names)
        for n in ast.walk(tree)
    )
    assert imports_it, "engine.py must import app.obs.failure_taxonomy."
    assert "failure_taxonomy.observe(" in src, (
        "engine.py must call failure_taxonomy.observe() so provider failures are "
        "classified live (Phase 1 #5 wiring)."
    )


def test_observe_classifies_provider_errors_live():
    # Exercise the exact path the engine hits: a rate-limit ProviderError.
    fc = observe(ProviderError("HTTP 429 too many requests"), where="test")
    assert fc.id == "provider_rate_limit"
    assert fc.recovery.value == "cooldown_wait"

    # And a timeout funnels correctly.
    assert classify_exception(TimeoutError("completion timed out")).id == "generation_timeout"


def test_observe_is_fail_open():
    class Boom(Exception):
        def __str__(self):
            raise RuntimeError("nope")
    # Must not raise even when the exception itself misbehaves.
    fc = observe(Boom(), where="test")
    assert fc.id == "internal_error"


def test_engine_retry_gate_consults_the_recovery_plan():
    """The plan must DRIVE the retry, not just get logged. Before the fix the gate
    was `if not exc.retryable: raise` with the plan used only in a log line."""
    src = _ENGINE.read_text(encoding="utf-8")
    assert "_plan_vetoes_retry(_plan)" in src, (
        "the retry gate must consult the recovery plan, not only exc.retryable"
    )
    # …and both loops (streaming + non-streaming) must be gated.
    assert src.count("if not exc.retryable or _plan_vetoes_retry(_plan)") == 1
    assert src.count("if yielded or not exc.retryable or _plan_vetoes_retry(_plan)") == 1
    # The plan's backoff and its different-route directive are acted on.
    assert "_plan_backoff_s(_plan, _action)" in src
    assert "_recovery.wants_different_route(_action)" in src


def test_both_engine_loops_classify_provider_errors():
    """The STREAMING path (the live hot path) was never classified at all — only
    route_and_complete called observe()."""
    src = _ENGINE.read_text(encoding="utf-8")
    assert 'where="engine.route_retry"' in src
    assert 'where="engine.stream_retry"' in src
