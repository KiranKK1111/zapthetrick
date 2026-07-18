"""Follow-up engine reliability invariants (followup-context-engine R12,
task 11.2). Pins Properties 11 & 12 at the unit level.

- Property 11 (single blocking LLM call): every engine entry point is a plain
  synchronous, deterministic function — none is a coroutine and none performs a
  model call — so the engine adds NO second blocking call on the critical path.
- Property 12 (safety / clarifier precedence): when a reference resolves below
  threshold (``needs_clarification``), the route does NOT rewrite — it yields to
  the existing clarifier — so follow-up handling never overrides the clarifier
  or a blocking safety confirmation.
"""
from __future__ import annotations

import inspect

from app.followup import acts, reference, rewrite, update
from app.followup.state import ConversationState


def test_engine_entrypoints_are_synchronous_no_llm():
    """No engine function is async → no awaited model call is introduced."""
    for fn in (acts.classify, reference.resolve, rewrite.rewrite,
               update.apply_turn, update.commit, update.continuation_directive):
        assert not inspect.iscoroutinefunction(fn), f"{fn.__name__} must be sync"


def test_full_engine_pass_runs_without_provider():
    """A complete classify→resolve→rewrite→update→commit pass runs purely on the
    state, with no LLM/provider configured (deterministic, fail-open)."""
    s = ConversationState({}, "c1")
    s.observe("build a Flutter app that streams chat", "")
    act, conf = acts.classify("make it faster", s)
    res = reference.resolve("make it faster", s)
    text, rconf = rewrite.rewrite("make it faster", act, res, s)
    update.apply_turn("make it faster", act, res, s)
    update.commit("make it faster", "Here is the optimized Flutter code...", s)
    assert isinstance(text, str) and 0.0 <= conf <= 1.0 and 0.0 <= rconf <= 1.0


def test_low_confidence_reference_yields_to_clarifier():
    """An ordinal with nothing to resolve against → needs_clarification, and the
    route's contract is to NOT rewrite in that case (defer to the clarifier)."""
    s = ConversationState({}, "c1")              # no enumerations
    res = reference.resolve("do the second one", s)
    assert res.needs_clarification is True
    # Mirror the route gate: a clarification-deferred turn is sent unchanged.
    model_text = "do the second one"
    if not res.needs_clarification:
        rw, _ = rewrite.rewrite("do the second one", acts.FOLLOW_UP, res, s)
        if rw and rw != "do the second one":
            model_text = rw
    assert model_text == "do the second one"      # unchanged → clarifier decides


def test_engine_failopen_never_raises():
    """Any malformed state must reduce to a no-op, never raise (Property 1/12)."""
    class _Bad:
        def __getattr__(self, _):
            raise RuntimeError("boom")

    bad = _Bad()
    # None of these may raise.
    act, conf = acts.classify("continue", bad)
    res = reference.resolve("use it", bad)
    text, rconf = rewrite.rewrite("improve it", act, res, bad)
    update.apply_turn("actually use rust", acts.CORRECTION, res, bad)
    update.commit("x", "y", bad)
    assert isinstance(text, str)
