"""Tests for the deterministic intent/confidence pre-gate
(AnalysisOnIntentsAndConfidence). Pure functions — no LLM, no DB."""
from __future__ import annotations

from app.clarify.intent_pipeline import (
    ANSWER,
    CLARIFY,
    DEFER,
    INTENT_CHITCHAT,
    INTENT_CODE_GEN,
    INTENT_COMPARISON,
    INTENT_KNOWLEDGE,
    INTENT_PROJECT_BUILD,
    assess,
    detect_intent,
    extract_slots,
)


class TestDetectIntent:
    def test_greeting(self):
        assert detect_intent("hi") == INTENT_CHITCHAT
        assert detect_intent("thanks!") == INTENT_CHITCHAT

    def test_code_generation(self):
        assert detect_intent(
            "i want a program for reversing a given string in java using streams"
        ) == INTENT_CODE_GEN
        assert detect_intent("write a function to sort a list") == INTENT_CODE_GEN

    def test_project_build_beats_code_verb(self):
        assert detect_intent("build me a web app") == INTENT_PROJECT_BUILD
        assert detect_intent("create a todo application") == INTENT_PROJECT_BUILD

    def test_knowledge_and_comparison(self):
        assert detect_intent("explain how kafka works") == INTENT_KNOWLEDGE
        assert detect_intent(
            "compare flutter vs react native") == INTENT_COMPARISON

    def test_explain_existing_code_is_read_only_not_codegen(self):
        """Explaining pasted/existing code must NOT be treated as code-generation
        (which would fire a spurious "which language?" clarification)."""
        for turn in (
            "can you explain me this program",
            "explain this code",
            "review this function",
            "what does this do",
            "walk me through the following",
        ):
            assert detect_intent(turn) == INTENT_KNOWLEDGE, turn

    def test_generation_requests_still_codegen(self):
        """A real generation request (no explain verb, no pasted code) is
        unchanged — still code-gen so a missing language can be asked."""
        assert detect_intent("write a program to reverse a string") == INTENT_CODE_GEN
        assert detect_intent("write a function to sort a list") == INTENT_CODE_GEN

    def test_pasted_code_with_explain_answers_not_clarifies(self):
        """The exact screenshot regression: pasted code + 'explain' must ANSWER,
        never clarify for a language."""
        pasted = (
            "```python\n"
            "def is_valid(board):\n"
            "    for row in board:\n"
            "        if not check(row):\n"
            "            return False\n"
            "    return True\n"
            "```\n"
            "can you explain me this program"
        )
        a = assess(pasted)
        assert a.decision == ANSWER
        assert "language" not in a.missing_required


class TestExtractSlots:
    def test_language_and_constraints(self):
        s = extract_slots(
            "reverse a given string in java using streams")
        assert s["language"] == "java"
        assert s["operation"] == "reverse"
        assert any("streams" in c for c in s["constraints"])

    def test_framework_vs_language(self):
        s = extract_slots("build a todo app in react with a node backend")
        assert s["framework"] == "react"
        assert s["language"] == "node"   # node is the non-framework tech token
        assert s["has_tech"]

    def test_platform(self):
        s = extract_slots("make a CLI tool")
        assert s["platform"] in ("cli", "command-line", "command line")

    def test_recent_window_supplies_tech(self):
        s = extract_slots("build the dashboard", recent="we use Django")
        assert s["has_tech"]


class TestAssessAnswerFirst:
    def test_reported_prompt_answers_directly(self):
        """The exact reported regression must now ANSWER, not clarify."""
        a = assess(
            "i want a program for reversing a given string in java using streams"
        )
        assert a.decision == ANSWER
        assert a.answerable
        assert not a.missing_required
        assert a.confidence >= 0.85
        assert "language" in a.suppressed

    def test_snippet_without_language_asks(self):
        # A code request with NO language now asks which language instead of
        # silently defaulting (e.g. to Python).
        a = assess("write a program to reverse a string")
        assert a.decision == CLARIFY
        assert "language" in a.missing_required

    def test_code_with_language_answers(self):
        a = assess("reverse a string in java using streams")
        assert a.decision == ANSWER
        assert "language" not in a.missing_required

    def test_code_with_framework_answers(self):
        # A framework implies a language → no need to ask.
        a = assess("write a component in react to list users")
        assert a.decision == ANSWER

    def test_code_language_known_pref_answers(self):
        a = assess("write a program to reverse a string",
                   known_prefs={"language": "python"})
        assert a.decision == ANSWER

    def test_document_without_format_asks(self):
        a = assess("can you document this")
        assert a.decision == CLARIFY
        assert "doc_format" in a.missing_required

    def test_document_with_format_answers(self):
        a = assess("document this as a pdf")
        assert a.decision == ANSWER

    def test_named_doc_format_never_asks_archive(self):
        # A named DOCUMENT format must never be mis-routed to the archive
        # "which format?" ask (the semantic classifier used to map "pdf
        # document" → archive). It should answer directly.
        from app.clarify.intent_pipeline import INTENT_DOCS
        a = assess("get me a pdf document")
        assert a.intent == INTENT_DOCS
        assert "archive_format" not in a.missing_required
        assert a.decision == ANSWER

    def test_generic_document_request_asks_format(self):
        # "get me a document for this" with no format named → ask which format.
        a = assess("get me a document for this")
        assert a.decision == CLARIFY
        assert "doc_format" in a.missing_required

    def test_compress_without_format_asks(self):
        from app.clarify.intent_pipeline import INTENT_ARCHIVE
        a = assess("compress this")
        assert a.intent == INTENT_ARCHIVE
        assert a.decision == CLARIFY
        assert "archive_format" in a.missing_required

    def test_compressed_file_of_project_asks(self):
        a = assess("get me the compressed file of the whole project")
        assert a.decision == CLARIFY
        assert "archive_format" in a.missing_required

    def test_compress_with_format_answers(self):
        a = assess("zip up the project")
        assert a.decision == ANSWER  # 'zip' names the format
        a2 = assess("give me a 7z of the whole project")
        assert a2.decision == ANSWER

    def test_build_zip_library_is_not_archive_intent(self):
        from app.clarify.intent_pipeline import INTENT_ARCHIVE
        # "build a zip library" is a code task, not a compress-this request.
        a = assess("write a python function to zip a folder")
        assert a.intent != INTENT_ARCHIVE

    def test_open_ended_build_clarifies(self):
        a = assess("build me a web app")
        assert a.decision == CLARIFY
        assert "language_or_framework" in a.missing_required
        assert a.confidence < 0.7
        assert a.strategy == "plan"

    def test_build_with_tech_answers(self):
        a = assess("build me a web app in React with a Node backend")
        assert a.decision == ANSWER
        assert not a.missing_required
        assert "framework" in a.suppressed

    def test_known_pref_satisfies_build_requirement(self):
        a = assess("build the dashboard", known_prefs={"language": "python"})
        assert a.decision == ANSWER
        assert "language" in a.suppressed

    def test_greeting_answers(self):
        a = assess("hello")
        assert a.decision == ANSWER
        assert a.intent == INTENT_CHITCHAT

    def test_knowledge_answers(self):
        a = assess("explain how a hashmap works")
        assert a.decision == ANSWER
        assert a.answerable

    def test_vague_request_is_ambiguous(self):
        a = assess("build me something")
        assert a.ambiguity > 0.0
        assert a.decision in (CLARIFY, DEFER)

    def test_confidence_in_range(self):
        for prompt in ("hi", "reverse a string in python", "build an app",
                       "explain recursion", "compare go and rust"):
            a = assess(prompt)
            assert 0.0 <= a.confidence <= 1.0


class TestConfidenceComposition:
    def test_specific_beats_vague(self):
        specific = assess("reverse a string in java using streams").confidence
        vague = assess("build me something").confidence
        assert specific > vague

    def test_named_tech_raises_build_confidence(self):
        with_tech = assess("build a web app in django").confidence
        without = assess("build a web app").confidence
        assert with_tech > without


class TestCodeRequestRequiresLanguage:
    """The 'ask which language when a code request names none, skip when it
    does' behavior that drives Supervisor.clarify_block (never-race)."""

    def test_no_language_program_request_requires_language(self):
        for prompt in (
            "can you give me a program for finding the 3rd non repeated character",
            "write a program to reverse a linked list",
            "can you give me a program for fibonacci",
            "implement a rate limiter",
        ):
            a = assess(prompt)
            assert a.decision == CLARIFY, prompt
            assert "language" in a.missing_required, prompt

    def test_named_language_answers_without_asking(self):
        for prompt in (
            "write a python program to reverse a linked list",
            "implement binary search in Java",
            "reverse a string in typescript",
            "give me a C++ program for fibonacci",
        ):
            a = assess(prompt)
            assert a.decision == ANSWER, prompt
            assert "language" not in a.missing_required, prompt


class TestArchiveIntentRobustness:
    """Archive intent isn't tied to one keyword (e.g. 'download') — it covers
    verbs, the bare 'archive' noun, and 'as a single file' phrasings, while
    NOT misfiring on definitional questions or code tasks."""

    def test_archive_phrasings_ask_format_when_unspecified(self):
        for prompt in (
            "get me the archive of this project",
            "i want the archive",
            "can I get an archive",
            "export the project",
            "bundle everything up",
            "give me the project archive",
            "make an archive of the code",
            "give me everything as a single file",
            "archive it",
        ):
            a = assess(prompt)
            assert detect_intent(prompt) == "archive", prompt
            assert a.decision == CLARIFY and "archive_format" in a.missing_required, prompt

    def test_named_format_produces_without_asking(self):
        for prompt in (
            "get me the archive of this project as zip",
            "send me the whole thing as a 7z",
            "download the project as zip",
        ):
            a = assess(prompt)
            assert detect_intent(prompt) == "archive", prompt
            assert "archive_format" not in a.missing_required, prompt

    def test_not_archive_for_questions_and_code(self):
        for prompt in (
            "what is a zip code",
            "what are zip codes",
            "how does compression work",
            "which is better zip or tar",
            "write a python function to zip two lists",
            "write a program to reverse a list",
        ):
            assert detect_intent(prompt) != "archive", prompt


class TestSemanticExemplarsRegexParity:
    """The regex fallback (used when the embedder is cold) must still classify
    the archive phrasings the semantic exemplars cover, so behavior is stable
    regardless of embedder readiness."""

    def test_regex_fallback_covers_archive_phrasings(self):
        for prompt in (
            "get me the archive of this project",
            "i want the archive",
            "export the project",
            "bundle everything up for me",
            "give me the project archive",
            "package it all up",
        ):
            # detect_intent is the deterministic fallback path.
            assert detect_intent(prompt) == "archive", prompt
