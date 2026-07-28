"""Stage-4 §3.8 Component F — deliverable decision engine.

The taxonomy stays DETERMINISTIC + explicit-only (no auto-file on a fuzzy
signal). This adds, behind `documents.deliverable_engine`: a confidence score, a
borderline OFFER (artifact-by-nature content, no explicit ask → stay inline +
offer a one-tap file), and a multi-deliverable picker set. An analysis verb
("summarize the contract") never offers — it's an answer ABOUT the artifact.
"""
from __future__ import annotations

import pytest

from app.documents import intent as I
from app.documents.intent import ArtifactIntent as AI


@pytest.fixture(autouse=True)
def _deterministic_taxonomy(monkeypatch):
    """This file asserts the DETERMINISTIC, explicit-only taxonomy (see the
    module docstring). `detect._semantic_doc_request` consults embedding gates
    when an embedder is loaded, and those legitimately resolve fuzzier phrasings
    ("create documentation for this") to an artifact intent — so the assertions
    below flipped depending on whether an EARLIER test in the session happened to
    warm the embedder. Pin the deterministic path so this file is order-
    independent and measures what it claims. (The semantic path is exercised by
    the document-intent gate tests; and per routes_agents.py, this classifier is
    observability only — triage, not this verdict, decides whether a file is
    generated.)"""
    from app.semantics import gates
    monkeypatch.setattr(gates, "_enabled", lambda: False)
    yield


@pytest.fixture
def engine_on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.documents, "deliverable_engine", True, raising=False)
    yield


def _c(text, **kw):
    return I.classify_artifact_intent(text, **kw)


# --------------------------------------------------------------------------- #
class TestBackwardCompatOff:
    def test_borderline_is_silent_chat_when_off(self):
        d = _c("write a cover letter for this job")
        assert d.intent is AI.CHAT
        assert d.offer_artifact is None       # no offer while the engine is off

    def test_explicit_unchanged(self):
        assert _c("export this as a pdf").intent is AI.ARTIFACT_ONLY
        assert _c("summarize this").intent is AI.CHAT

    def test_documentation_no_format_stays_chat(self):
        # The explicit-only anti-false-doc guard is preserved.
        assert _c("create documentation for this").intent is AI.CHAT


class TestBorderlineOffer:
    def test_cover_letter_offers_inline_file(self, engine_on):
        d = _c("write a cover letter for this job")
        assert d.intent is AI.CHAT            # still inline — explicit-only kept
        assert d.offer_artifact == "pdf"
        assert d.confidence == 0.5

    def test_presentation_offers_pptx(self, engine_on):
        assert _c("build a presentation about kafka").offer_artifact == "pptx"

    def test_blog_post_offers_md(self, engine_on):
        assert _c("write a blog post on rust").offer_artifact == "md"

    def test_need_a_resume_offers(self, engine_on):
        assert _c("I need a resume").offer_artifact == "pdf"


class TestAnalysisGuard:
    def test_summarize_contract_no_offer(self, engine_on):
        d = _c("summarize this contract for me")
        assert d.intent is AI.CHAT and d.offer_artifact is None

    def test_review_resume_no_offer(self, engine_on):
        assert _c("review my resume").offer_artifact is None

    def test_explain_the_invoice_no_offer(self, engine_on):
        assert _c("explain the invoice line items").offer_artifact is None


class TestExplicitStillWins:
    def test_explicit_format_is_artifact(self, engine_on):
        assert _c("give me a resume as a docx").intent is AI.ARTIFACT_ONLY

    def test_high_confidence_on_explicit(self, engine_on):
        assert _c("export this as a pdf").confidence == 0.9


class TestMultiDeliverable:
    def test_two_deliverables_detected(self):
        got = I.detect_deliverables("make me a resume and a cover letter")
        assert [g["noun"] for g in got] == ["resume", "cover letter"]
        assert all(g["format"] == "pdf" for g in got)

    def test_single_deliverable_no_picker(self):
        assert I.detect_deliverables("make me a resume") == []

    def test_analysis_request_no_picker(self):
        assert I.detect_deliverables("summarize the resume and the report") == []

    def test_mixed_formats(self):
        got = I.detect_deliverables("a resume and a presentation on my career")
        fmts = {g["noun"]: g["format"] for g in got}
        assert fmts == {"resume": "pdf", "presentation": "pptx"}


class TestDeliverableNouns:
    def test_cover_letter_suppresses_bare_letter(self):
        got = I.deliverable_nouns("a cover letter please")
        assert got == [("cover letter", "pdf")]      # not also a bare "letter"

    def test_none_for_plain_answer(self):
        assert I.deliverable_nouns("what is a hashmap") == []


class TestSerialization:
    def test_as_dict_carries_new_fields(self, engine_on):
        d = _c("write a cover letter")
        js = d.as_dict()
        assert js["offer_artifact"] == "pdf"
        assert js["confidence"] == 0.5
