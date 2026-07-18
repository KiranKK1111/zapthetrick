"""Tests for Tool/Capability Reliability Scores (roadmap Phase 5 #11),
including the wiring into the tool executor's dispatch loop — both the
*recording* half and the *steering* half (degraded tools get deprioritised).
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib

import pytest

from app.tools import reliability as rel


@pytest.fixture(autouse=True)
def _clean():
    rel.reset()
    yield
    rel.reset()


def test_unknown_tool_is_neutral():
    assert rel.reliability("never_seen") == 0.5
    assert not rel.is_degraded("never_seen")


def test_reliability_converges_to_success_rate():
    for _ in range(90):
        rel.record("t", True)
    for _ in range(10):
        rel.record("t", False)
    # 90/100 with Laplace ≈ 0.89
    assert 0.87 <= rel.reliability("t") <= 0.90


def test_degraded_needs_enough_history():
    rel.record("t", False)  # 1 failure only
    assert not rel.is_degraded("t")  # too little history to condemn
    for _ in range(5):
        rel.record("t", False)
    assert rel.is_degraded("t")


def test_rank_prefers_reliable():
    for _ in range(10):
        rel.record("good", True)
    for _ in range(10):
        rel.record("bad", False)
    assert rel.rank(["bad", "good"]) == ["good", "bad"]


def test_snapshot_shape():
    rel.record("t", True); rel.record("t", False)
    snap = rel.snapshot()
    assert snap["t"]["attempts"] == 2
    assert 0.0 <= snap["t"]["reliability"] <= 1.0


def test_record_is_fail_open():
    rel.record(None, True)  # type: ignore[arg-type]  — must not raise


def test_executor_wires_reliability():
    # The tool executor must record success/failure per dispatch.
    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "tools" / "executor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports_rel = any(
        isinstance(n, ast.ImportFrom) and n.module == "app.tools"
        and any(a.name == "reliability" for a in n.names)
        for n in ast.walk(tree)
    )
    assert imports_rel, "executor must import app.tools.reliability"
    assert "reliability.record(tool.name, True)" in src
    assert "reliability.record(tool.name, False)" in src


# ── reliability STEERS dispatch (not just measures it) ───────────────────────
#
# The executor's real choice points: (a) the catalog it shows the model, and
# (b) which of the model's chosen calls win the `max_tools` slots. Both are
# exercised end-to-end through `run_relevant_tools` with a stubbed LLM.

@pytest.fixture
def toolbox(monkeypatch):
    """Register throwaway tools + stub the classifier LLM. Yields a helper that
    runs the executor with a scripted set of model-chosen calls and returns
    (executed tool names, the catalog the model was shown)."""
    from app.core import llm_client
    from app.tools import registry
    from app.tools.executor import run_relevant_tools

    ran: list[str] = []
    seen: dict[str, str] = {}
    added: list[str] = []

    def add(name: str, *, fails: bool = False):
        async def handler(query: str = ""):
            ran.append(name)
            if fails:
                raise RuntimeError(f"{name} is broken")
            return f"{name}:ok"
        registry.register(registry.Tool(
            name=name, description=f"{name} description",
            input_schema={"type": "object",
                          "properties": {"query": {"type": "string"}}},
            handler=handler))
        added.append(name)

    def run(picks: list[str], *, max_tools: int = 1):
        async def fake_complete(messages, **kw):
            seen["prompt"] = messages[0]["content"]
            return json.dumps({"calls": [{"name": p, "arguments": {"query": "q"}}
                                         for p in picks]})
        monkeypatch.setattr(llm_client.llm, "complete", fake_complete)
        ran.clear()
        results = asyncio.run(run_relevant_tools(
            "a question", allow=set(added), max_tools=max_tools))
        return ran[:], seen.get("prompt", ""), results

    try:
        yield add, run
    finally:
        for n in added:
            registry._registry.pop(n, None)


def _degrade(name: str, n: int = 6):
    for _ in range(n):
        rel.record(name, False)


def _marked(prompt: str) -> list[str]:
    """Catalog lines carrying the [unreliable] marker (the prompt's static
    preamble mentions the marker too — that's not a flagged tool)."""
    return [ln for ln in prompt.splitlines()
            if ln.startswith("- ") and "[unreliable]" in ln]


def _bless(name: str, n: int = 6):
    for _ in range(n):
        rel.record(name, True)


def test_degraded_tool_deprioritised_when_healthy_alternative_exists(toolbox):
    add, run = toolbox
    add("flaky_search")
    add("solid_search")
    _degrade("flaky_search")
    _bless("solid_search")

    # The model picks the flaky one FIRST, but only one slot is available:
    # reliability must hand that slot to the healthy alternative.
    ran, prompt, results = run(["flaky_search", "solid_search"], max_tools=1)
    assert ran == ["solid_search"]
    assert [r["tool"] for r in results] == ["solid_search"]

    # …and the model was told: healthy tool first, flaky one marked.
    assert prompt.index("solid_search") < prompt.index("flaky_search")
    assert "- flaky_search — flaky_search description [unreliable]" in prompt
    assert [ln for ln in _marked(prompt)] == [
        "- flaky_search — flaky_search description [unreliable]"]


def test_degraded_tool_still_runs_when_it_is_the_only_option(toolbox):
    # Never hard-block the only path to a capability.
    add, run = toolbox
    add("only_tool")
    _degrade("only_tool")
    ran, prompt, _ = run(["only_tool"], max_tools=1)
    assert ran == ["only_tool"]
    # Nothing healthier exists, so don't badmouth it to the model either.
    assert not _marked(prompt)


def test_degraded_tool_runs_when_both_fit_under_the_cap(toolbox):
    add, run = toolbox
    add("flaky2")
    add("solid2")
    _degrade("flaky2")
    _bless("solid2")
    # Two slots, two calls → both run; the healthy one just goes first.
    ran, _, _ = run(["flaky2", "solid2"], max_tools=2)
    assert ran == ["solid2", "flaky2"]


def test_tool_with_no_history_is_not_penalised(toolbox):
    add, run = toolbox
    add("brand_new")
    add("also_new")
    # Zero history for both → the model's own order is preserved, no demotion.
    ran, prompt, _ = run(["brand_new", "also_new"], max_tools=1)
    assert ran == ["brand_new"]
    assert not _marked(prompt)
    assert prompt.index("brand_new") < prompt.index("also_new")


def test_new_tool_beats_a_degraded_one(toolbox):
    add, run = toolbox
    add("proven_bad")
    add("untested")
    _degrade("proven_bad")
    # Unknown (0.5) outranks known-bad (<0.4) — but only because it's degraded,
    # not because the tool is new.
    ran, _, _ = run(["proven_bad", "untested"], max_tools=1)
    assert ran == ["untested"]


def test_dispatch_survives_a_broken_reliability_store(toolbox, monkeypatch):
    add, run = toolbox
    add("t_a")
    add("t_b")
    _degrade("t_a")

    def boom(*a, **kw):
        raise RuntimeError("reliability store exploded")

    monkeypatch.setattr(rel, "is_degraded", boom)
    monkeypatch.setattr(rel, "rank", boom)
    monkeypatch.setattr(rel, "reliability", boom)

    # Fail-open: dispatch behaves exactly as it did before reliability existed —
    # the model's first pick runs, in the model's order.
    ran, prompt, results = run(["t_a", "t_b"], max_tools=1)
    assert ran == ["t_a"]
    assert [r["tool"] for r in results] == ["t_a"]
    assert not _marked(prompt)


def test_dispatch_records_outcomes_while_steering(toolbox):
    add, run = toolbox
    add("boom_tool", fails=True)
    run(["boom_tool"], max_tools=1)
    snap = rel.snapshot()["boom_tool"]
    assert snap["attempts"] == 1 and snap["reliability"] < 0.5
