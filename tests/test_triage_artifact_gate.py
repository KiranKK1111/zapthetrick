"""Document-gating precision guard (issue: docs appearing unrequested).

Under `documents.explicit_only` (the default), the triage classifier only trusts
the LLM's `document:true` when the message also names a downloadable artifact
(format / file / download / export). This pins that guard: it must block
in-chat "summarize this" / "give me a report" (no artifact token) while catching
explicit "as a PDF" / "export as excel" phrasings the strict regex may miss.
"""
from __future__ import annotations

from app.chat.triage import _mentions_artifact


def test_plain_answer_requests_have_no_artifact_token():
    for msg in (
        "give me a report",
        "summarize this",
        "explain quicksort",
        "what is recursion",
        "write a report on climate change",  # in-chat report, no file token
    ):
        assert _mentions_artifact(msg) is False, msg


def test_answer_seeking_phrases_are_not_artifacts():
    """Regression (2026-07-14): 'Can I have solution for a different problem …'
    generated an UNREQUESTED PDF because the fuzzy semantic artifact gate fired
    on 'solution'/'can I have'. The gate is deterministic-only now, so an answer/
    solution request (no file word) never counts as an artifact — no matter how
    warm the embedder is."""
    for msg in (
        "Can I have solution for different problem statement related to this",
        "can i have a solution",
        "give me the approach",
        "solve this problem",
        "can you give me another example",
        "what is the output for this",
    ):
        assert _mentions_artifact(msg) is False, msg


def test_explicit_file_requests_have_an_artifact_token():
    for msg in (
        "give me a report as a pdf",
        "export this as excel",
        "make a word document",
        "can you zip the project",
        "put it in a csv",
        "download this as a file",
    ):
        assert _mentions_artifact(msg) is True, msg


def test_document_retrieval_followups_reproduce_the_file():
    """A doc-location follow-up must re-produce the downloadable card instead of
    letting the model confabulate ("I already provided it / use Download")."""
    from app.documents.detect import explicit_doc_request
    for msg in (
        "where is the document",
        "where's the pdf",
        "show me the file",
        "send me the document",
        "download the document",
        "resend the word document",
    ):
        assert explicit_doc_request(msg)[0] is True, msg


def test_retrieval_does_not_overfire_on_code_or_summaries():
    from app.documents.detect import explicit_doc_request
    for msg in (
        "give me a report",           # in-chat report, not a file
        "summarize this",
        "create a file to store data",  # coding, not a document
        "how do I open a pptx in python",
        "show me the code",
    ):
        assert explicit_doc_request(msg)[0] is False, msg
