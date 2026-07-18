"""Claude-style clarify-first for vague deliverable requests.

A first-prompt deliverable ask with no subject/content yet — "can you get me
a document", "get me the archive of this project", "give me a source-code
file" — must produce a NATURAL, VARIED clarifying question (like Claude), not
a hallucinated deliverable and not a robotic fixed string. This pins:
  * the directive that drives that behavior (per kind), and
  * the `assess()` oracle the route uses to decide vague-vs-specified.
"""
from __future__ import annotations

from app.api.routes_agents import _empty_target_directive
from app.clarify.intent_pipeline import (ANSWER, CLARIFY, INTENT_ARCHIVE,
                                         INTENT_CODE_GEN, INTENT_DOCS, assess,
                                         detect_intent)


# ── The directive (what makes the model ASK instead of inventing) ────────────

def test_directive_forbids_invention_and_asks_for_each_kind():
    for kind in ("docs", "archive", "code"):
        d = _empty_target_directive(kind)
        low = d.lower()
        # It must tell the model to CLARIFY…
        assert "clarify" in low
        assert "question" in low
        # …and explicitly NOT fabricate a deliverable (the old hallucination).
        assert "do not invent" in low or "not invent" in low
        assert "fabricate" in low
        # …in its own natural words, not a template (so regens vary).
        assert "your own words" in low or "not a template" in low


def test_directive_examples_are_kind_specific():
    assert "resume" in _empty_target_directive("docs").lower()
    assert "zip" in _empty_target_directive("archive").lower()
    assert "script" in _empty_target_directive("code").lower()


def test_unknown_kind_defaults_to_document():
    # Any unexpected kind falls back to the generic document phrasing.
    assert "document" in _empty_target_directive("").lower()


# ── The detection oracle: vague deliverables → CLARIFY, specified → ANSWER ────

def test_vague_document_request_is_underspecified():
    for q in ("can you get me a document", "can you give me a document"):
        a = assess(q, [], {})
        assert detect_intent(q) == INTENT_DOCS
        assert a.decision == CLARIFY, q


def test_archive_of_nonexistent_project_is_archive_intent():
    q = "get me the archive of this project"
    assert detect_intent(q) == INTENT_ARCHIVE
    assert assess(q, [], {}).decision == CLARIFY


def test_vague_code_file_request_is_underspecified():
    for q in ("give me a source code file", "can you give me a program"):
        assert detect_intent(q) == INTENT_CODE_GEN
        assert assess(q, [], {}).decision == CLARIFY, q


def test_fully_specified_code_request_answers_directly():
    # A request that already names language + task must NOT be treated as an
    # empty-target clarify — it answers.
    q = "write a python program to reverse a linked list"
    assert detect_intent(q) == INTENT_CODE_GEN
    assert assess(q, [], {}).decision == ANSWER


def test_assess_accepts_nonstring_recent():
    # REGRESSION: the route passed `recent` as a LIST; assess did
    # `recent.strip()` and raised AttributeError, which the route's broad
    # try/except swallowed — so empty-target detection silently never fired
    # on a first prompt and a stray PDF was generated for the clarifying
    # question. A non-empty list must NOT throw and must still classify.
    a = assess("get me the document", ["get me the document"], {})
    assert a.decision == CLARIFY
    # And with a genuine prior-context list.
    a2 = assess("get me the document", ["earlier turn", "another"], {})
    assert a2.decision == CLARIFY


def test_first_prompt_document_is_empty_target():
    # The exact reported case: first prompt "get me the document" with the
    # route's real (non-empty) recent list must resolve to an under-specified
    # docs turn → empty-target → no document generated.
    recent = ["get me the document"]  # route builds [current] on a first turn
    a = assess("get me the document", " ".join(recent[:-1]), {})
    assert detect_intent("get me the document") == INTENT_DOCS
    assert a.decision == CLARIFY  # → _suppress_empty_target True → _doc_sources None
