"""Stage-4 §3.5 — patch-based follow-up edit turn-loop wiring.

`_try_patch_answer` detects an edit against the code just produced, asks for
str_replace patches (structured), applies them, and returns the revised answer
— or None to fall through to a full mesh regeneration. The `code_patch` module
itself is covered by test_code_patch.py; here we test the route wiring's
branches with `structured` + the semantic gate stubbed.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import app.api.routes_agents as R
from app.llm.structured import StructuredResult


def _run(coro):
    return asyncio.run(coro)


_PRIOR = [
    {"role": "user", "content": "write a python add function"},
    {"role": "assistant",
     "content": "Here you go:\n\n```python\ndef add(a, b):\n    return a + b\n```"},
]


@pytest.fixture
def _on(monkeypatch):
    from app.chat import code_patch as cp
    monkeypatch.setattr(cp, "enabled", lambda: True)
    # Force the edit-intent decision True (the gate is exercised in its own test).
    monkeypatch.setattr(cp, "is_edit_request", lambda text, *, has_prior_code: True)
    # Keep re-verify out of the sandbox for this unit test.
    from app.codeintel import code_verify as cv
    monkeypatch.setattr(cv, "enabled", lambda: False)


def _stub_structured(monkeypatch, obj):
    async def fake(schema, messages, **kw):
        return StructuredResult(obj=obj, errors=[] if obj is not None else ["x"])
    monkeypatch.setattr(R, "structured", fake, raising=False)
    # `structured` is imported inside the function; patch the source module too.
    import app.core.structured as cs
    monkeypatch.setattr(cs, "structured", fake)


class TestPatchTurn:
    def test_applies_patch_and_returns_revised(self, monkeypatch, _on):
        _stub_structured(monkeypatch, {"patches": [
            {"old_str": "return a + b", "new_str": "return a + b  # sum"}]})
        out = _run(R._try_patch_answer("add a comment", _PRIOR, "c1"))
        assert out is not None
        assert "# sum" in out and "```python" in out

    def test_stale_target_falls_through(self, monkeypatch, _on):
        # old_str not present in the prior code → apply fails → None (full regen).
        _stub_structured(monkeypatch, {"patches": [
            {"old_str": "NONEXISTENT", "new_str": "x"}]})
        assert _run(R._try_patch_answer("change it", _PRIOR, "c1")) is None

    def test_invalid_structured_falls_through(self, monkeypatch, _on):
        _stub_structured(monkeypatch, None)
        assert _run(R._try_patch_answer("change it", _PRIOR, "c1")) is None

    def test_no_prior_code_returns_none(self, monkeypatch, _on):
        prior = [{"role": "assistant", "content": "just prose, no code here"}]
        assert _run(R._try_patch_answer("change it", prior, "c1")) is None

    def test_not_an_edit_returns_none(self, monkeypatch, _on):
        from app.chat import code_patch as cp
        monkeypatch.setattr(cp, "is_edit_request",
                            lambda text, *, has_prior_code: False)
        assert _run(R._try_patch_answer("what is a monad?", _PRIOR, "c1")) is None

    def test_feature_off_returns_none(self, monkeypatch):
        from app.chat import code_patch as cp
        monkeypatch.setattr(cp, "enabled", lambda: False)
        assert _run(R._try_patch_answer("add a comment", _PRIOR, "c1")) is None
