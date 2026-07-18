"""Empty-session export/archive guard.

Pins the fix for two first-prompt chat bugs:
* "get me the archive of this project" used to trigger an LLM-invented
  clarify card ("Which project should be archived?" with location options)
  — nonsensical when the chat has nothing to package.
* "export the project" used to produce a fully hallucinated "Project
  Export" document for a project that doesn't exist.

The Clarifier now DECLINES outright on an archive/export turn when the
session has no target (no prior code, no attachments, nothing pasted), and
the route suppresses file generation + injects a guidance directive instead.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import clarifier as clarifier_mod
from app.agents.clarifier import KEY_CLARIFY_META, ClarifierAgent
from app.blackboard.board import Blackboard
from app.blackboard.schema import KEY_QUESTION


def _run_clarifier(question: str, extras: dict, monkeypatch) -> Blackboard:
    """Run the ClarifierAgent with the LLM gate BOOBY-TRAPPED: reaching it
    fails the test, proving the deterministic guard decided first."""

    async def _no_llm(*a, **k):
        raise AssertionError("LLM clarify gate must not run for this turn")

    monkeypatch.setattr(clarifier_mod.llm, "complete", _no_llm)
    board = Blackboard()
    board.write(KEY_QUESTION, question)
    board.write("extras", extras)
    asyncio.run(ClarifierAgent().run(board))
    return board


@pytest.mark.parametrize("question", [
    "get me the archive of this project",
    "export the project",
    "give me a zip of the project",
    "download the project",
])
def test_archive_request_with_empty_session_declines(question, monkeypatch):
    board = _run_clarifier(question, {}, monkeypatch)
    assert board.get("clarifying_questions") == []
    meta = board.get(KEY_CLARIFY_META) or {}
    assert "nothing to archive" in (meta.get("reason") or "").lower()


@pytest.mark.parametrize("question,extras", [
    # A rendering follow-up ("in a tabular format") describes how the inline
    # answer should look — never a document deliverable.
    ("can you get me in a tabular format", {"has_prior_content": True}),
    ("show it as a table", {"has_prior_content": True}),
    ("give me this as bullet points", {"has_prior_content": True}),
    # A statement the intent classifier can mislabel as code_generation must
    # not fire the "which language?" card (regression 2026-07-16).
    ("I don't want pin and section", {}),
    ("what is the difference between monolith and microservices", {}),
])
def test_no_false_clarification_on_followups_and_statements(
        question, extras, monkeypatch):
    board = _run_clarifier(question, extras, monkeypatch)
    assert board.get("clarifying_questions") == [], (
        f"{question!r} must be answered, not clarified")


def test_archive_request_with_prior_code_still_asks_format(monkeypatch):
    extras = {
        "prior_messages": [
            {"role": "assistant", "content": "```python\nprint('hi')\n```"},
        ],
        "has_prior_code": True,
        "has_prior_content": True,
    }
    board = _run_clarifier(
        "get me the archive of this project", extras, monkeypatch)
    qs = board.get("clarifying_questions") or []
    assert qs, "with real content the archive-format ask must survive"
    text = str(qs).lower()
    assert "format" in text or "zip" in text


def test_maybe_clarify_passes_content_signals(monkeypatch):
    """The upload-stream path (quality.maybe_clarify) must not ask the
    invented 'which project?' question on an empty session either."""

    async def _no_llm(*a, **k):
        raise AssertionError("LLM clarify gate must not run for this turn")

    monkeypatch.setattr(clarifier_mod.llm, "complete", _no_llm)
    from app.chat.quality import maybe_clarify

    qs = asyncio.run(maybe_clarify("get me the archive of this project", []))
    assert qs == []
