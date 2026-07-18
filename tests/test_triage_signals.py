"""Folded triage signals — topic_shift + read_only (roadmap #12 / Phase C)."""
from __future__ import annotations

import asyncio

import pytest

from app.chat import triage as tri
from app.clarify.intent_pipeline import is_read_only
from app.followup.acts import is_topic_shift


@pytest.fixture(autouse=True)
def _deterministic_signals(monkeypatch):
    """Pin the DETERMINISTIC layer. is_topic_shift / is_read_only are now
    SEMANTIC-first (topic_shift / read_only gates), which is warm in the full
    suite and returns valid-but-different verdicts than the cue lists these tests
    pin. The gate mechanism is covered in test_semantic_gates; here we test the
    fallback + the folding, so disable the embedding gates."""
    import app.semantics.gates as _g
    monkeypatch.setattr(_g, "matches", lambda *a, **k: None)


def test_is_read_only_detects_explain_existing():
    assert is_read_only("explain this function")
    assert is_read_only("what does this code do")
    assert is_read_only("review the following implementation")


def test_is_read_only_false_for_generation():
    assert not is_read_only("write a function to reverse a string")
    assert not is_read_only("build me a todo app")


def test_signals_topic_shift():
    ts, ro = tri._signals("new topic: tell me about Rust")
    assert ts is True
    assert is_topic_shift("new topic: tell me about Rust")


def test_signals_read_only():
    ts, ro = tri._signals("explain this snippet")
    assert ro is True and ts is False


def test_signals_plain_turn():
    ts, ro = tri._signals("write a REST API in Go")
    assert ts is False and ro is False


def test_triage_dataclass_defaults():
    t = tri.Triage()
    assert t.topic_shift is False and t.read_only is False


def test_triage_greeting_fastpath_carries_signals():
    # greetings skip the LLM call entirely — a good no-network test that the
    # signals ride the returned object.
    out = asyncio.run(tri.triage("hello"))
    assert out.difficulty == "trivial"
    assert out.topic_shift is False and out.read_only is False


def test_triage_greeting_with_topic_shift_signal():
    # "forget that" is a topic-shift cue; still trivial + no LLM call.
    out = asyncio.run(tri.triage("forget that"))
    assert out.topic_shift is True


def _triage_with_llm(monkeypatch, doc_flag: bool):
    """Run triage with the LLM leg stubbed to return `document: doc_flag`,
    so we can test the deterministic per-turn/inheritance gating."""
    async def fake_complete(messages, model=None, options=None):
        return '{"difficulty": "standard", "document": %s, "format": "pdf"}' % (
            "true" if doc_flag else "false")
    from app.core import llm_client
    monkeypatch.setattr(llm_client.llm, "complete", fake_complete)


def test_file_intent_is_per_turn_no_stickiness(monkeypatch):
    """A prior '.py file' request must NOT force a file onto a later, fresh
    program request (allow_recent_doc defaults False)."""
    _triage_with_llm(monkeypatch, doc_flag=False)
    out = asyncio.run(tri.triage(
        "write a program to reverse a linked list",
        recent="give me a .py file for a sorting algorithm"))
    assert out.wants_document is False


def test_file_intent_inherited_only_on_clarification_answer(monkeypatch):
    """When THIS turn is a clarification answer, the prior file request carries
    (allow_recent_doc=True) so a clarifier round-trip still makes the file."""
    _triage_with_llm(monkeypatch, doc_flag=False)
    out = asyncio.run(tri.triage(
        "python",  # answering "which language?"
        recent="give me a source code file for a sorting algorithm",
        allow_recent_doc=True))
    assert out.wants_document is True


def test_new_topic_with_explicit_file_request_makes_file(monkeypatch):
    """A brand-new request that ITSELF names a file → file, even with no
    inheritance (per-turn detection on the current text)."""
    _triage_with_llm(monkeypatch, doc_flag=False)
    out = asyncio.run(tri.triage(
        "write a python program to reverse a list and give me the .py file",
        recent="explain how TCP works"))
    assert out.wants_document is True
