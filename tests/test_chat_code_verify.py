"""Stage-4 §3.1 Component B — verify-while-streaming for chat code.

Covers the tree-sitter pre-gate (`app/codeintel/pregate.py`) and the reusable
chat verify lane (`app/codeintel/code_verify.py`): fence extraction, the sticky
language ladder, the should-verify gate, and the verify_stream orchestration
(pass / repair-and-swap / no-code / cancel / timeout) — all with a stubbed
`verify_and_maybe_repair`, so no sandbox or network is touched.
"""
from __future__ import annotations

import asyncio

import pytest

from app.codeintel import code_verify as cv
from app.codeintel import pregate as pg


def _run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    stages, result = [], None
    async for kind, payload in agen:
        if kind == "stage":
            stages.append(payload)
        elif kind == "result":
            result = payload
    return stages, result


PY_CODE = "```python\ndef add(a, b):\n    return a + b\n```"


# --------------------------------------------------------------------------- #
# Tree-sitter pre-gate
# --------------------------------------------------------------------------- #
class TestPregate:
    def test_clean_python_ok(self):
        assert pg.parse_ok("def f(x):\n    return x + 1", "python") == (True, None)

    def test_broken_python_flagged(self):
        ok, err = pg.parse_ok("def f(x)\n    return x", "python")  # missing colon
        assert ok is False and "syntax error" in err

    def test_unknown_language_abstains(self):
        assert pg.parse_ok("!!!nonsense!!!", "brainfuck") == (True, None)

    def test_empty_is_ok(self):
        assert pg.parse_ok("", "python") == (True, None)

    def test_language_mapping(self):
        assert pg.ts_language("Python 3") == "python"
        assert pg.ts_language("C#") == "csharp"
        assert pg.ts_language("js") == "javascript"
        assert pg.ts_language("brainfuck") is None

    def test_looks_like_code_rejects_prose(self):
        assert pg.looks_like_code("just some prose sentence here", "python") is False

    def test_looks_like_code_accepts_real_code(self):
        assert pg.looks_like_code("def f():\n    return 1", "python") is True

    def test_looks_like_code_abstains_unknown_lang(self):
        assert pg.looks_like_code("anything", "brainfuck") is True


# --------------------------------------------------------------------------- #
# Lane helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_extract_primary_code(self):
        text = f"Here you go:\n\n{PY_CODE}\n\nHope that helps."
        got = cv.extract_primary_code(text)
        assert got is not None
        code, label = got
        assert label == "python" and "def add" in code

    def test_extract_skips_one_liner(self):
        assert cv.extract_primary_code("```python\nx = 1\n```") is None

    def test_extract_none_when_no_fence(self):
        assert cv.extract_primary_code("no code at all here") is None

    def test_resolve_language_ladder(self):
        assert cv.resolve_language(explicit="go", fence_label="python") == "go"
        assert cv.resolve_language(fence_label="python", sticky="rust") == "python"
        assert cv.resolve_language(sticky="rust") == "rust"
        assert cv.resolve_language() is None

    def test_plan_returns_language_for_real_code(self):
        assert cv.plan(PY_CODE, question="write it in python") == "python"

    def test_plan_none_without_code(self):
        assert cv.plan("just prose, nothing to run") is None

    def test_plan_none_for_prose_in_fence(self):
        text = "```python\nthis is not code just words in a fence line two\n```"
        assert cv.plan(text) is None

    def test_should_verify_needs_language(self):
        assert cv.should_verify(PY_CODE, language_label=None) is False
        assert cv.should_verify(PY_CODE, language_label="python") is True

    def test_enabled_default_off(self):
        assert cv.enabled() is False


# --------------------------------------------------------------------------- #
# The verify lane
# --------------------------------------------------------------------------- #
def _stub_verify(monkeypatch, *, verdict, fixed, stages=(), hang=False):
    import app.codeintel.solution_verify as sv

    async def fake(problem, answer, label, examples=None, max_repairs=1,
                   on_stage=None, min_difficulty=None):
        for s in stages:
            if on_stage:
                await on_stage(s)
        if hang:
            await asyncio.sleep(10)
        return (verdict, fixed)
    monkeypatch.setattr(sv, "verify_and_maybe_repair", fake)


class TestVerifyStream:
    def test_pass_appends_verdict_only(self, monkeypatch):
        _stub_verify(monkeypatch, verdict="\n\n---\n✅ Compiled & ran", fixed=None,
                     stages=["Reading examples", "Running in sandbox"])

        async def go():
            return await _collect(cv.verify_stream(
                "write add()", PY_CODE, language_label="python"))
        stages, res = _run(go())
        assert set(stages) >= {"Reading examples", "Running in sandbox"}
        assert res.ran is True
        assert res.fixed_code is None
        assert "✅ Compiled & ran" in res.delta
        assert "```" not in res.delta            # no new code block on a pass
        assert res.updated_text.endswith("✅ Compiled & ran")

    def test_failure_swaps_in_fixed_block(self, monkeypatch):
        _stub_verify(monkeypatch,
                     verdict="\n\n---\n⚠️ Fixed a runtime error",
                     fixed="def add(a, b):\n    return a + b  # fixed")

        async def go():
            return await _collect(cv.verify_stream(
                "write add()", PY_CODE, language_label="python"))
        _stages, res = _run(go())
        assert res.ran is True
        assert res.fixed_code is not None
        assert "```python" in res.delta          # repaired block appended
        assert "Fixed a runtime error" in res.delta

    def test_no_code_skips(self, monkeypatch):
        _stub_verify(monkeypatch, verdict="should not run", fixed=None)

        async def go():
            return await _collect(cv.verify_stream(
                "q", "just prose no code", language_label="python"))
        _stages, res = _run(go())
        assert res.ran is False
        assert res.updated_text == "just prose no code"

    def test_cancellation_leaves_answer_untouched(self, monkeypatch):
        _stub_verify(monkeypatch, verdict="x", fixed=None, hang=True)
        killed = []

        async def go():
            return await _collect(cv.verify_stream(
                "q", PY_CODE, language_label="python",
                is_cancelled=lambda: True,
                cancel_sandbox=lambda: killed.append(1)))
        _stages, res = _run(go())
        assert res.ran is False
        assert res.updated_text == PY_CODE
        assert killed == [1]                      # sandbox exec was killed

    def test_timeout_is_honest(self, monkeypatch):
        _stub_verify(monkeypatch, verdict="x", fixed=None, hang=True)

        async def go():
            return await _collect(cv.verify_stream(
                "q", PY_CODE, language_label="python", deadline_s=0.15))
        _stages, res = _run(go())
        assert res.ran is False
        assert "timed out" in res.suffix
