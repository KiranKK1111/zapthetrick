"""Phase 0 — Artifact Intent Taxonomy (Document Generation roadmap).

Pins the deterministic classification of a turn's desired outcome. Two invariants
matter most: (1) it reproduces DocuementGeneration.md's example table, and (2) it
NEVER classifies an ordinary answer-seeking turn as an artifact — the guard that
keeps unrequested PDFs from appearing (the recurring bug this session)."""
from __future__ import annotations

import asyncio
import json

import pytest

import app.api.routes_agents as ra
from app.documents.intent import (
    ArtifactIntent, PlannerDecision, classify_artifact_intent,
    SOURCE_EXISTING, SOURCE_LAST_RESPONSE, SOURCE_NEW,
)

C = classify_artifact_intent


# ── helpers for the route's re-delivery path (no DB in the unit suite) ───────
class _Row:
    """Stand-in for a `generated_documents` row (app/documents/store.py)."""

    def __init__(self, content_md: str, doc_format: str, title: str = ""):
        self.content_md = content_md
        self.doc_format = doc_format
        self.title = title


class _FakeSession:
    """`get_session_factory()` returns a factory; `factory()` is an async CM."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _drain(agen) -> list[str]:
    async def _run():
        return [frame async for frame in agen]
    return asyncio.run(_run())


def _events(frames: list[str]) -> list[tuple[str, dict]]:
    """Parse SSE frames into (event, payload) pairs."""
    out: list[tuple[str, dict]] = []
    for f in frames:
        lines = f.strip().splitlines()
        out.append((lines[0].split("event: ", 1)[1],
                    json.loads(lines[1].split("data: ", 1)[1])))
    return out


@pytest.fixture(autouse=True)
def _deterministic_detectors(monkeypatch):
    """Pin the DETERMINISTIC layer. classify_artifact_intent defers to
    explicit_doc_request BY DESIGN (so its verdict always agrees with triage —
    no split-brain), and that detector has an embedding semantic tail which is
    active in the full suite (warm embedder) but silent in isolation. That
    state-dependence made these tests flaky; disable the tail so we test the
    deterministic classification. The semantic recall is triage's concern and is
    covered there."""
    import app.documents.detect as _d
    monkeypatch.setattr(_d, "_semantic_doc_request", lambda _t: None)


class TestDocExampleTable:
    """The mapping table from the source document (Step 1), adapted to the app's
    explicit-only policy where it must (see test_deviations_from_doc)."""

    @pytest.mark.parametrize("text,prior_art,prior_content,expected", [
        ("Explain Kafka", False, True, ArtifactIntent.CHAT),
        ("Generate PDF", False, True, ArtifactIntent.ARTIFACT_ONLY),
        ("Convert above into Word", True, True, ArtifactIntent.UPDATE_EXISTING),
        ("Export this project", False, True, ArtifactIntent.ARTIFACT_ONLY),
        ("Give me architecture", False, True, ArtifactIntent.CHAT),
    ])
    def test_table(self, text, prior_art, prior_content, expected):
        assert C(text, has_prior_artifact=prior_art,
                 has_prior_content=prior_content).intent == expected


class TestNeverOverGenerates:
    """No file for a turn that names no format/export verb — regardless of prior
    content. Pins every false-doc phrase this session surfaced."""

    @pytest.mark.parametrize("text", [
        "Can I have solution for different problem statement related to this",
        "can you give me some more details on it",
        "what is the output for this",
        "give me a report",             # in-chat report, not a file
        "create documentation for this",  # no format named → chat (app policy)
        "explain quicksort",
        "why does this happen",
    ])
    def test_chat(self, text):
        d = C(text, has_prior_artifact=False, has_prior_content=True)
        assert d.intent == ArtifactIntent.CHAT
        assert d.wants_artifact is False

    def test_deviation_from_doc_is_intentional(self):
        # The source doc maps "Create documentation for this" → ANSWER_AND_ARTIFACT.
        # DETERMINISTICALLY we return CHAT: no format/export named ⇒ no file,
        # matching ZapTheTrick's explicit-only policy (unrequested-PDF fix). The
        # shared semantic tail (triage's `document_request` gate) MAY upgrade it
        # live — that's fine: the classifier and triage move together, so there's
        # never a split-brain where one generates and the other doesn't.
        assert C("create documentation for this",
                 has_prior_content=True).intent == ArtifactIntent.CHAT


class TestArtifactKinds:
    def test_artifact_only_reuses_prior_answer(self):
        d = C("Generate PDF", has_prior_content=True)
        assert d.intent == ArtifactIntent.ARTIFACT_ONLY
        assert d.reuse_response is True and d.requires_llm is False
        assert d.source == SOURCE_LAST_RESPONSE
        assert d.artifact_type == "pdf"

    def test_answer_and_artifact_authors_new_content(self):
        d = C("generate a document on kafka basics", has_prior_content=False)
        assert d.intent == ArtifactIntent.ANSWER_AND_ARTIFACT
        assert d.reuse_response is False and d.requires_llm is True
        assert d.source == SOURCE_NEW

    @pytest.mark.parametrize("text,fmt", [
        ("convert this to excel", "xlsx"),
        ("give me this as a word document", "docx"),
        ("zip the project", "zip"),
        ("export this project", "zip"),
    ])
    def test_artifact_only_formats(self, text, fmt):
        d = C(text, has_prior_content=True)
        assert d.intent == ArtifactIntent.ARTIFACT_ONLY and d.artifact_type == fmt


class TestExistingArtifactOps:
    def test_download_existing_no_regen(self):
        d = C("where is the pdf", has_prior_artifact=True, has_prior_content=True)
        assert d.intent == ArtifactIntent.DOWNLOAD_EXISTING
        assert d.reuse_response is True and d.requires_llm is False
        assert d.source == SOURCE_EXISTING

    def test_update_existing_needs_prior_artifact(self):
        # With a prior artifact → UPDATE; without one the same words are a
        # first-time ARTIFACT_ONLY (nothing to update yet).
        assert C("convert the above into Word", has_prior_artifact=True,
                 has_prior_content=True).intent == ArtifactIntent.UPDATE_EXISTING
        assert C("convert this into a pdf", has_prior_artifact=False,
                 has_prior_content=True).intent == ArtifactIntent.ARTIFACT_ONLY

    def test_download_requires_prior_artifact(self):
        # "download this data" with nothing produced yet isn't a re-delivery.
        assert C("download this", has_prior_artifact=False,
                 has_prior_content=True).intent == ArtifactIntent.ARTIFACT_ONLY


class TestDownloadExistingIsReEmittedNotRegenerated:
    """BUG (2026-07-14): a DOWNLOAD_EXISTING turn was classified perfectly and
    then IGNORED — the route ran the full LLM generation anyway and authored a
    NEW document the user already had. The route now short-circuits on the
    planner's OWN signal (`reuse_response` + `requires_llm=False`) and re-emits
    the stored artifact with no model call."""

    def test_planner_signal_drives_the_reuse(self):
        d = C("where is the pdf", has_prior_artifact=True, has_prior_content=True)
        assert d.reuse_response is True and d.requires_llm is False
        assert ra._reuses_existing_artifact(d) is True

    @pytest.mark.parametrize("text,prior_art", [
        ("Explain Kafka", True),                          # CHAT
        ("Generate PDF", False),                          # ARTIFACT_ONLY
        ("generate a document on kafka basics", False),   # ANSWER_AND_ARTIFACT
        ("convert the above into Word", True),            # UPDATE_EXISTING
    ])
    def test_every_other_intent_still_generates(self, text, prior_art):
        d = C(text, has_prior_artifact=prior_art, has_prior_content=True)
        assert d.intent != ArtifactIntent.DOWNLOAD_EXISTING
        assert ra._reuses_existing_artifact(d) is False

    def test_archive_redelivery_is_left_to_the_zip_fast_path(self):
        d = PlannerDecision(ArtifactIntent.DOWNLOAD_EXISTING, "zip",
                            source=SOURCE_EXISTING, reuse_response=True,
                            requires_llm=False)
        assert ra._reuses_existing_artifact(d) is False

    def test_redelivers_the_stored_artifact_without_calling_the_llm(
            self, monkeypatch):
        """The whole point: a re-delivery emits the download card for the
        EXISTING document — no LLM, no new content."""
        from app.core import llm_client

        async def _no_llm(*a, **k):
            raise AssertionError("a re-delivery must never call the LLM")

        monkeypatch.setattr(llm_client.llm, "complete", _no_llm)
        # No DB in the unit suite: the persist is best-effort, the card still ships.
        monkeypatch.setattr(ra, "get_session_factory", lambda: None)

        d = C("resend the document", has_prior_artifact=True,
              has_prior_content=True)
        assert d.intent == ArtifactIntent.DOWNLOAD_EXISTING
        art = {"content": "# Kafka\n\nThe stored document.", "format": "pdf",
               "title": "Kafka"}
        evts = _events(_drain(ra._redeliver_artifact("conv-1", art, d)))

        assert evts[0] == ("meta", {"doc_pending": "pdf"})
        streamed = "".join(p["text"] for k, p in evts if k == "token")
        assert streamed == art["content"]          # the EXISTING doc, verbatim
        kind, done = evts[-1]
        assert kind == "done"
        assert done["document"] == {
            "document": True, "format": "pdf", "formats": ["pdf"],
            "artifact_intent": "DOWNLOAD_EXISTING", "reuse_response": True,
            "redelivered": True,
        }

    def test_reads_the_versioned_store_first(self, monkeypatch):
        import app.documents.store as store

        async def _latest(_session, _sid):
            return _Row("# Stored\n\nfrom the document store.", "docx",
                        "Stored")

        monkeypatch.setattr(store, "latest_for_session", _latest)
        monkeypatch.setattr(ra, "get_session_factory", lambda: _FakeSession)
        got = asyncio.run(ra._existing_artifact("conv-1"))
        assert got["content"].startswith("# Stored")
        assert got["format"] == "docx"

    def test_named_format_this_turn_wins_over_the_stored_one(self, monkeypatch):
        import app.documents.store as store

        async def _latest(_session, _sid):
            return _Row("# Stored\n\nbody.", "pdf", "Stored")

        monkeypatch.setattr(store, "latest_for_session", _latest)
        monkeypatch.setattr(ra, "get_session_factory", lambda: _FakeSession)
        got = asyncio.run(ra._existing_artifact("conv-1", want_format="docx"))
        assert got["format"] == "docx"

    def test_falls_back_to_the_prior_artifact_message(self, monkeypatch):
        # The store is best-effort (record_generation is fail-open), so a
        # conversation can have an artifact message with no store row.
        monkeypatch.setattr(ra, "get_session_factory", lambda: None)
        got = asyncio.run(ra._existing_artifact(
            "conv-1", fallback={"content": "# Old doc\n\nbody.",
                                "format": "xlsx"}))
        assert got["format"] == "xlsx" and got["content"].startswith("# Old doc")

    def test_no_prior_artifact_degrades_to_normal_generation(self, monkeypatch):
        # NOTHING on file → None → the route falls through to today's behavior
        # (generate). It must never invent or error.
        monkeypatch.setattr(ra, "get_session_factory", lambda: None)
        assert asyncio.run(ra._existing_artifact("conv-1")) is None
        assert asyncio.run(
            ra._existing_artifact("conv-1", fallback={"content": "  "})) is None

    def test_store_error_degrades_to_normal_generation(self, monkeypatch):
        import app.documents.store as store

        async def _boom(_session, _sid):
            raise RuntimeError("database is down")

        monkeypatch.setattr(store, "latest_for_session", _boom)
        monkeypatch.setattr(ra, "get_session_factory", lambda: _FakeSession)
        assert asyncio.run(ra._existing_artifact("conv-1")) is None
        # …but a thread-local fallback still saves the re-delivery.
        assert asyncio.run(ra._existing_artifact(
            "conv-1", fallback={"content": "# Old\n\nbody."}))["format"] == "pdf"

    def test_an_archive_on_file_is_never_re_emitted_as_a_document(
            self, monkeypatch):
        monkeypatch.setattr(ra, "get_session_factory", lambda: None)
        assert asyncio.run(ra._existing_artifact(
            "conv-1", fallback={"content": "```py\nx=1\n```",
                                "format": "zip"})) is None


class TestPlannerDecisionShape:
    def test_as_dict_and_wants_artifact(self):
        d = C("Generate PDF", has_prior_content=True)
        assert isinstance(d, PlannerDecision)
        js = d.as_dict()
        assert set(js) == {"intent", "artifact_type", "source",
                           "reuse_response", "requires_llm"}
        assert js["intent"] == "ARTIFACT_ONLY"
        assert C("Explain Kafka").wants_artifact is False

    def test_empty_is_chat(self):
        assert C("").intent == ArtifactIntent.CHAT
