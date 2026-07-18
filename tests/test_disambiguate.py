"""LLM intent disambiguation for low-confidence turns (gap G4)."""
from __future__ import annotations

import asyncio

from app.understanding import disambiguate as dis


def _run(coro):
    return asyncio.run(coro)


def test_match_exact_label():
    assert dis._match("code_generation") == "code_generation"


def test_match_tolerates_extra_text():
    assert dis._match("This is debugging.") == "debugging"
    assert dis._match("Intent: comparison") == "comparison"


def test_match_none_when_no_label():
    assert dis._match("banana") is None
    assert dis._match("") is None


def test_disambiguate_returns_label():
    async def fake(_text):
        return "project_build"
    assert _run(dis.disambiguate_intent("build me an app", complete_fn=fake)) \
        == "project_build"


def test_disambiguate_none_on_empty_text():
    async def fake(_text):
        raise AssertionError("should not be called")
    assert _run(dis.disambiguate_intent("", complete_fn=fake)) is None


def test_disambiguate_fail_open():
    async def boom(_text):
        raise RuntimeError("model down")
    assert _run(dis.disambiguate_intent("q", complete_fn=boom)) is None


def test_disambiguate_none_on_unmatched_reply():
    async def fake(_text):
        return "I'm not sure"
    assert _run(dis.disambiguate_intent("q", complete_fn=fake)) is None


def test_enabled_reads_config(monkeypatch):
    from app.core import config_loader as cl

    class _SI:
        llm_disambiguation = True
    monkeypatch.setattr(cl.cfg, "semantic_intent", _SI(), raising=False)
    assert dis.enabled() is True
