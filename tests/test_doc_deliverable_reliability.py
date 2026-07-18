"""Deliverable-reliability regressions (user report 2026-07-08: "document /
archive generation not happening for all requests; sometimes no downloadable
file at all"). Pins the broadened deterministic detector (produce-a-document
phrasings, package/bundle archives, how-to and library-name guards) and the
sandbox parse-verification for rendered documents."""
from __future__ import annotations

import pytest

from app.documents.detect import explicit_doc_request


class TestDetectorRecall:
    """Explicit produce-a-file phrasings must fire deterministically — the
    triage LLM leg is a fallback, not a requirement."""

    @pytest.mark.parametrize("text,fmt", [
        ("generate a document on kafka basics", "pdf"),
        ("create a document about our api design", "pdf"),
        ("make me a document for the onboarding process", "pdf"),
        ("i want this as a downloadable document", "pdf"),
        ("can you make this downloadable", "pdf"),
        ("export this conversation", "pdf"),
        ("generate the report as a document", "pdf"),
        ("put this in a document", "pdf"),
        ("i need a soft copy of this", "pdf"),
        ("package the code for download", "zip"),
        ("bundle this up so i can download it", "zip"),
        ("zip the code", "zip"),
        ("archive the project", "zip"),
        ("give me a pdf document", "pdf"),
        ("make a word document of the summary", "docx"),
        ("how do i get this as a pdf", "pdf"),
        # Text formats resolve to THEIR format, not pdf (group-index bug:
        # the retrieval pattern read the VERB as the noun → always pdf).
        ("give me this as json", "json"),
        ("export this as a json file", "json"),
        ("give me a txt file of this", "txt"),
        ("as a text file please", "txt"),
        ("give me this in markdown", "md"),
        ("as a markdown file", "md"),
        ("give me a csv of the data", "csv"),
        ("make a csv file from this table", "csv"),
        ("send me the xlsx", "xlsx"),
        ("resend the pdf", "pdf"),
        # Article forms: "in a <format>".
        ("give me this in a pdf", "pdf"),
        ("put this in a txt", "txt"),
        ("summary in a spreadsheet please", "xlsx"),
        ("save it in word format", "docx"),
    ])
    def test_fires(self, text, fmt):
        det, got = explicit_doc_request(text)
        assert det, text
        assert got == fmt, (text, got)


class TestDetectorPrecision:
    """Ordinary questions that merely mention a format/document never fire."""

    @pytest.mark.parametrize("text", [
        "summarize the document",
        "what is in the document i uploaded",
        "how do i create a document in python-docx",
        "how do i compress files in python",
        "how do i zip files in java",
        "explain the document structure of mongodb",
        "create a python package that parses csv",
        "what does this code do",
        "write a login api in flask",
        "review my document below",
        "how do i generate a pdf in python using fpdf",
        "how do i parse json in python",
        "return json from the api",
        "read a csv with pandas",
        "what is markdown",
        "in a word, yes it will scale",
    ])
    def test_clean(self, text):
        det, _ = explicit_doc_request(text)
        assert not det, text


class TestSandboxDocVerify:
    """Rendered documents are re-opened by their own parser in the sandbox."""

    def test_valid_docx_verifies(self):
        import asyncio
        from app.documents.generators import render_document
        from app.verify.doc_verify import verify_document_bytes
        data, _, ext = render_document("# Title\n\nHello world", "docx")
        meta = asyncio.run(verify_document_bytes(data, ext))
        assert meta is not None
        # "verified" when the sandbox ran; "skipped" only if unavailable.
        assert meta["status"] in ("verified", "skipped")

    def test_corrupt_docx_fails(self):
        import asyncio
        from app.verify.doc_verify import verify_document_bytes
        meta = asyncio.run(verify_document_bytes(b"this is not a docx",
                                                 "docx"))
        assert meta is not None
        assert meta["status"] in ("failed", "skipped")

    def test_undriven_format_returns_none(self):
        import asyncio
        from app.verify.doc_verify import verify_document_bytes
        assert asyncio.run(verify_document_bytes(b"hello", "txt")) is None
        assert asyncio.run(verify_document_bytes(b"hello", "md")) is None

    def test_json_and_csv_drivers(self):
        import asyncio
        from app.verify.doc_verify import verify_document_bytes
        ok = asyncio.run(verify_document_bytes(b'{"a": 1}', "json"))
        assert ok is not None and ok["status"] in ("verified", "skipped")
        bad = asyncio.run(verify_document_bytes(b"{not json", "json"))
        assert bad is not None and bad["status"] in ("failed", "skipped")
        rows = asyncio.run(verify_document_bytes(b"a,b\n1,2\n", "csv"))
        assert rows is not None and rows["status"] in ("verified", "skipped")


class TestClarificationFormatAnswer:
    """A bare format chip the user taps when asked "which format?"
    ("Format: Word (.docx)") is invisible to explicit_doc_request. The Word→ZIP
    bug (2026-07-13): with prior code in the chat the archive path silently won
    and delivered a zip. format_answer parses the chip so a chosen DOCUMENT
    format is honored and overrides packaging."""

    @pytest.mark.parametrize("text,fmt", [
        ("Format: Word (.docx)", "docx"),
        ("Word (.docx)", "docx"),
        ("PDF", "pdf"),
        ("Format: PDF", "pdf"),
        ("Excel (.xlsx)", "xlsx"),
        ("Format: PowerPoint (.pptx)", "pptx"),
        ("a zip file", "zip"),
        ("7-zip", "7z"),
        ("markdown", "md"),
    ])
    def test_parses_chip_answers(self, text, fmt):
        from app.documents.detect import format_answer
        assert format_answer(text) == fmt

    @pytest.mark.parametrize("text", [
        "",
        "give me this in a document",          # a request, not a bare answer
        "let us discuss the pdf spec and how word processors handle text "
        "across many long paragraphs of prose here so nothing misfires",  # prose
    ])
    def test_ignores_non_answers(self, text):
        from app.documents.detect import format_answer
        assert format_answer(text) is None

    def test_bare_answer_invisible_to_produce_detector(self):
        """Documents WHY format_answer exists: the produce-a-document detectors
        miss a bare chip answer, so the format choice would be dropped."""
        from app.documents.detect import (
            explicit_doc_request, explicit_doc_formats, format_answer,
        )
        assert explicit_doc_request("Format: Word (.docx)") == (False, None)
        assert explicit_doc_formats("Format: Word (.docx)") == []
        assert format_answer("Format: Word (.docx)") == "docx"

    @pytest.mark.parametrize("text,named", [
        ("give me this as a word document", True),
        ("as a PDF", True),
        ("i want an excel sheet", True),
        ("give me this in a document", False),   # generic → defaults, NOT named
        ("put this in a document", False),
        ("can you make this downloadable", False),
    ])
    def test_mentions_format_distinguishes_named_from_defaulted(self, text, named):
        """The progress label must not claim a format the user never named:
        mentions_format is True only when a real format token appears, unlike
        explicit_doc_formats which silently defaults a generic request to pdf."""
        from app.documents.detect import mentions_format
        assert mentions_format(text) is named


class TestNamedFormatRecall:
    """Direct (non-clarification) named-format requests must resolve to the
    format the user named — regressions found 2026-07-13 where "into"/"convert
    to <format>" phrasings silently produced nothing (and would fall back to a
    default). The format the user CHOSE must drive generation."""

    @pytest.mark.parametrize("text,fmt", [
        ("put this into an excel spreadsheet", "xlsx"),
        ("put this into excel", "xlsx"),
        ("put this into a spreadsheet", "xlsx"),
        ("put this into a powerpoint", "pptx"),
        ("convert this to excel", "xlsx"),
        ("turn this into a pdf", "pdf"),
        ("export this to powerpoint", "pptx"),
        ("save this as a word document", "docx"),
    ])
    def test_into_and_convert_phrasings(self, text, fmt):
        from app.documents.detect import explicit_doc_formats
        assert explicit_doc_formats(text) == [fmt]

    @pytest.mark.parametrize("text", [
        "convert the list to json",       # data-processing, not a file request
        "parse this to csv",
        "convert the string to a number",
        "in a word, yes that works",       # idiom — no produce verb
        "how do I convert a dataframe to excel in python",  # code how-to guard
        "can you explain word embeddings",
    ])
    def test_does_not_oversweep_code_or_idioms(self, text, monkeypatch):
        # These pin the DETERMINISTIC regex layer (the part these gaps live in).
        # The embedding SEMANTIC TAIL is a separate authority that, once warm,
        # can classify a borderline idiom like "in a word …" as a doc request —
        # a state-dependent verdict that would make this flaky. Disable it so we
        # test exactly the layer the fix touches.
        import app.documents.detect as _d
        monkeypatch.setattr(_d, "_semantic_doc_request", lambda _t: None)
        assert _d.explicit_doc_formats(text) == []
