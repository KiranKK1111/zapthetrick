"""Data-driven clarifier behavior suite (AechitectureLikeClaude.md).

116 scenarios extracted verbatim from the scenario catalog into
tests/data/clarifier_scenarios.json — each is (prompt, should_clarify,
has_artifact). The deterministic pre-gate's contract against them:

  * expected CLARIFY  → decision in (CLARIFY, DEFER)  (DEFER → the LLM gate
    still gets the chance to ask; only a confident deterministic ANSWER is a
    hard under-ask)
  * expected ANSWER   → decision in (ANSWER, DEFER)   (a deterministic CLARIFY
    here is a hard over-ask — the "unnecessary clarification" failure mode)

Over-asks must be ZERO (never interrogate a user who already said what they
want). Under-asks are capped at the known accepted misses listed below.
"""
from __future__ import annotations

import io
import json
import pathlib

import pytest

from app.clarify.intent_pipeline import ANSWER, CLARIFY, assess

_DATA = pathlib.Path(__file__).parent / "data" / "clarifier_scenarios.json"
_SCENARIOS = json.load(io.open(_DATA, encoding="utf-8"))

# Accepted misses (documented, not bugs):
#   119 "Build a RAG system." — RAG is a named (partial) stack; answering with
#       stated assumptions matches the catalog's own core principle.
_ACCEPTED_UNDER_ASK = {119}


def test_catalog_no_over_ask_and_bounded_under_ask():
    over, under = [], []
    for s in _SCENARIOS:
        d = assess(s["prompt"], has_artifact=bool(s.get("has_artifact"))).decision
        if s["should_clarify"]:
            if d == ANSWER and s["id"] not in _ACCEPTED_UNDER_ASK:
                under.append((s["id"], s["prompt"]))
        else:
            if d == CLARIFY:
                over.append((s["id"], s["prompt"]))
    assert not over, f"unnecessary clarifications (over-ask): {over}"
    assert not under, f"missed required clarifications (under-ask): {under}"


# ---- The core product scenarios, pinned individually --------------------

_PASTED = "explain this code\n\ndef f(x):\n    return x * 2\nprint(f(3))\nimport sys"


# expect: ANSWER / CLARIFY exact, or None = "anything but a deterministic
# CLARIFY" (ANSWER and DEFER both honor the no-unnecessary-questions rule;
# DEFER can occur when the semantic classifier is stubbed out in CI).
@pytest.mark.parametrize("name,prompt,has_artifact,expect,expect_missing", [
    ("explain pasted code answers", _PASTED, False, None, None),
    ("bare doc request asks format",
     "get me a document for this", False, CLARIFY, "doc_format"),
    ("named doc format answers",
     "get me a word document for this", False, None, None),
    ("attached image ask answers",
     "what is wrong here in my code", True, None, None),
    ("program without language asks",
     "write a program to reverse a string", False, CLARIFY, "language"),
    ("program with language answers",
     "write a java program to reverse a string", False, ANSWER, None),
    ("project without stack asks",
     "build an e-commerce application", False, CLARIFY,
     "language_or_framework"),
    ("project with stack answers",
     "build an e-commerce app using react, spring boot and postgresql",
     False, ANSWER, None),
    ("knowledge question is not vague", "what is a hash map?", False,
     None, None),
    ("concrete operation answers",
     "reverse a linked list in python", False, ANSWER, None),
    ("fix-my-code without code asks for it",
     "fix my code", False, CLARIFY, "artifact"),
    ("unit tests without code ask for it",
     "write unit tests", False, CLARIFY, "artifact"),
    ("vague deploy asks",
     "deploy my application", False, CLARIFY, None),
])
def test_core_scenarios(name, prompt, has_artifact, expect, expect_missing):
    a = assess(prompt, has_artifact=has_artifact)
    if expect is None:
        assert a.decision != CLARIFY, (
            f"{name}: unnecessary clarification (missing="
            f"{a.missing_required})")
    else:
        assert a.decision == expect, (
            f"{name}: got {a.decision} (missing={a.missing_required})")
    if expect_missing:
        assert expect_missing in a.missing_required, (
            f"{name}: missing_required={a.missing_required}")


def test_attachment_suppresses_artifact_ask():
    # Same prompt: without an artifact → ask for it; with one → answerable.
    bare = assess("fix my code")
    assert bare.decision == CLARIFY and "artifact" in bare.missing_required
    attached = assess("fix my code", has_artifact=True)
    assert attached.decision != CLARIFY or \
        "artifact" not in attached.missing_required


def test_pasted_stack_trace_counts_as_artifact():
    a = assess("My Spring Boot API returns 500. Here is the stack trace.")
    assert "artifact" not in a.missing_required


def test_language_not_asked_for_attached_code():
    a = assess("Apply a Factory Pattern to this code.", has_artifact=True)
    assert a.decision != CLARIFY, a.missing_required
