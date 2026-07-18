"""Phase 5 persistence — REAL DB round-trip integration tests.

Unlike the unit tests (which pin the pure logic), these exercise the actual
save/commit/load path against Postgres — the gap flagged during the build. Each
test builds its OWN async engine bound to its event loop (the app's global
engine is loop-bound and singleton, which fights per-test loops), monkeypatches
`get_session_factory` so the self-contained `record_generation` uses it, seeds a
real `sessions` row for the FK, and cleans up via ON DELETE CASCADE.

Auto-SKIPPED when Postgres isn't reachable, so a DB-less CI run stays green.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.documents.lifecycle import merge_update
from app.documents.store import (
    latest_for_session, latest_version, list_for_session, list_versions,
    record_generation, save_version,
)
from storage.models import GeneratedDocument, Session


def _make_engine():
    """A fresh async engine configured EXACTLY like the app's (crucially the
    schema search_path GUC — without it the app's tables don't resolve)."""
    from storage.db import _build_url, _search_path
    return create_async_engine(
        _build_url(),
        connect_args={"server_settings": {"search_path": _search_path()}})


def _db_reachable() -> bool:
    async def _c() -> None:
        eng = _make_engine()
        try:
            async with eng.begin() as conn:
                await conn.execute(text("SELECT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(), reason="Postgres not reachable for integration tests")


def _run(monkeypatch, body) -> None:
    """Run ``body(sf, session_id)`` against a fresh engine + a seeded session row,
    with `get_session_factory` pointed at that engine; always cleans up."""
    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: sf)

    async def run() -> None:
        sid = uuid.uuid4()
        async with sf() as s:
            s.add(Session(id=sid, title="pytest-doc-store", type="chat"))
            await s.commit()
        try:
            await body(sf, sid)
        finally:
            async with sf() as s:
                obj = await s.get(Session, sid)
                if obj is not None:
                    await s.delete(obj)      # CASCADE removes its documents
                    await s.commit()
            await eng.dispose()

    asyncio.run(run())


def test_save_version_creates_and_chains(monkeypatch):
    async def body(sf, sid):
        async with sf() as s:
            r1 = await save_version(s, sid, "# Doc v1", title="Doc", fmt="pdf")
            await s.commit()
            key, v1 = r1.doc_key, r1.version
        assert v1 == 1
        async with sf() as s:
            r2 = await save_version(s, sid, "# Doc v2", doc_key=key)
            await s.commit()
            assert r2.version == 2 and r2.doc_key == key
        async with sf() as s:
            versions = await list_versions(s, key)
            assert [v.version for v in versions] == [1, 2]
            assert versions[0].content_md == "# Doc v1"
    _run(monkeypatch, body)


def test_latest_for_session_and_version(monkeypatch):
    async def body(sf, sid):
        async with sf() as s:
            r1 = await save_version(s, sid, "# A", title="A")
            await s.commit()
            key = r1.doc_key
        async with sf() as s:
            await save_version(s, sid, "# B", doc_key=key)
            await s.commit()
        async with sf() as s:
            assert (await latest_for_session(s, sid)).content_md == "# B"
            assert (await latest_version(s, key)).version == 2
    _run(monkeypatch, body)


def test_record_generation_self_contained(monkeypatch):
    async def body(sf, sid):
        key = await record_generation(sid, "# First\n\nbody", title="First",
                                      fmt="docx")
        assert key is not None
        async with sf() as s:
            rows = await list_versions(s, key)
            assert len(rows) == 1 and rows[0].doc_format == "docx"
            assert rows[0].title == "First"
    _run(monkeypatch, body)


def test_record_generation_chain_latest(monkeypatch):
    async def body(sf, sid):
        k1 = await record_generation(sid, "# Doc\n\nv1", title="Doc")
        # chain_latest → next version of the same document, no explicit key.
        k2 = await record_generation(sid, "# Doc\n\nv2 edited", title="Doc",
                                     chain_latest=True)
        assert k2 == k1
        async with sf() as s:
            rows = await list_versions(s, k1)
            assert [r.version for r in rows] == [1, 2]
            assert rows[1].content_md == "# Doc\n\nv2 edited"
    _run(monkeypatch, body)


def test_update_existing_merge_flow(monkeypatch):
    """The full UPDATE_EXISTING path: persist v1, then merge an edit and store
    the MERGED full document as v2 of the same doc_key."""
    prior = ("# System Design\n\n## Overview\n\nThe system.\n\n"
             "## Database\n\nUses MySQL.\n")

    async def body(sf, sid):
        k1 = await record_generation(sid, prior, title="System Design")
        # Load prior, merge this turn's edit (adds a Caching section), chain it.
        async with sf() as s:
            prev = await latest_for_session(s, sid)
        merged = merge_update(prev.content_md,
                              "Adding it:\n\n## Caching\n\nRedis for hot data.")
        k2 = await record_generation(sid, merged, title="System Design",
                                     chain_latest=True)
        assert k2 == k1
        async with sf() as s:
            v2 = await latest_version(s, k1)
            assert v2.version == 2
            # v2 is the FULL merged document, not just the edit.
            assert "Caching" in v2.content_md and "Redis" in v2.content_md
            assert "MySQL" in v2.content_md and "Overview" in v2.content_md
    _run(monkeypatch, body)


def test_list_for_session_newest_first(monkeypatch):
    async def body(sf, sid):
        await record_generation(sid, "# One", title="One")
        await record_generation(sid, "# Two", title="Two")
        async with sf() as s:
            rows = await list_for_session(s, sid)
            titles = [r.title for r in rows]
            assert "One" in titles and "Two" in titles
            # newest first
            assert rows[0].created_at >= rows[-1].created_at
    _run(monkeypatch, body)


def test_artifacts_endpoints_return_persisted(monkeypatch):
    """The inspection endpoints read back what generation persisted (async
    handlers called directly so they share this test's loop + factory)."""
    async def body(sf, sid):
        from app.api.routes_documents import (
            list_artifact_versions, list_document_artifacts)
        key = await record_generation(sid, "# Doc\n\nv1", title="Doc", fmt="pdf")
        await record_generation(sid, "# Doc\n\nv2 edited", title="Doc",
                                chain_latest=True)
        arts = await list_document_artifacts(session_id=str(sid))
        assert any(a["doc_key"] == str(key) for a in arts["artifacts"])
        vers = await list_artifact_versions(doc_key=str(key))
        assert [v["version"] for v in vers["versions"]] == [1, 2]
        assert vers["versions"][1]["content_md"] == "# Doc\n\nv2 edited"
    _run(monkeypatch, body)


def test_search_documents(monkeypatch):
    """Phase 6 cross-artifact search over real persisted documents."""
    async def body(sf, sid):
        from app.documents.graph import search_documents
        await record_generation(
            sid, "# Auth\n\nJWT authentication with OAuth2 flow.", title="Auth")
        await record_generation(
            sid, "# Billing\n\nStripe charges and invoices.", title="Billing")
        async with sf() as s:
            hits = await search_documents(s, "authentication", session_id=str(sid))
            assert any(h["title"] == "Auth" for h in hits)
            assert all("authentication" in h["snippet"].lower()
                       or h["title"] == "Auth" for h in hits)
            # A term in neither doc → no hits.
            assert await search_documents(s, "kubernetes", session_id=str(sid)) == []
    _run(monkeypatch, body)


def test_artifact_graph(monkeypatch):
    """Phase 6 relationship graph: version chains + sibling edges."""
    async def body(sf, sid):
        from app.documents.graph import build_artifact_graph
        k1 = await record_generation(sid, "# Doc A v1", title="A")
        await record_generation(sid, "# Doc A v2", title="A", chain_latest=True)
        await record_generation(sid, "# Doc B", title="B")   # a sibling document
        async with sf() as s:
            g = await build_artifact_graph(s, str(sid))
        assert len(g["documents"]) == 2                      # two doc_keys
        node_a = next(n for n in g["documents"] if n["doc_key"] == str(k1))
        assert node_a["latest_version"] == 2
        assert [v["version"] for v in node_a["versions"]] == [1, 2]
        assert len(g["edges"]) == 1 and g["edges"][0]["kind"] == "sibling"
    _run(monkeypatch, body)


def test_preferences_round_trip(monkeypatch):
    """Phase 7 doc-gen memory: save + reload document preferences (KV table)."""
    async def body(sf, sid):
        from app.documents.preferences import (
            get_preferences, save_preferences)
        prefs = {"default_format": "docx",
                 "branding": {"header": "ACME", "primary_color": "#4f46e5"}}
        assert await save_preferences(prefs) is True
        got = await get_preferences()
        assert got.get("default_format") == "docx"
        assert got["branding"]["header"] == "ACME"
        # Overwrite (upsert), not duplicate.
        assert await save_preferences({"default_format": "pdf"}) is True
        assert (await get_preferences()).get("default_format") == "pdf"
        # cleanup the KV row so the test is isolated.
        from sqlalchemy import delete
        from storage.models import LLMSetting
        async with sf() as s:
            await s.execute(delete(LLMSetting).where(LLMSetting.key == "doc_prefs"))
            await s.commit()
    _run(monkeypatch, body)


def test_cascade_delete_removes_documents(monkeypatch):
    async def body(sf, sid):
        key = await record_generation(sid, "# Ephemeral", title="E")
        async with sf() as s:
            assert len(await list_versions(s, key)) == 1
        # Delete the session; the FK CASCADE must remove its documents.
        async with sf() as s:
            await s.delete(await s.get(Session, sid))
            await s.commit()
        async with sf() as s:
            gone = (await s.execute(
                select(GeneratedDocument)
                .where(GeneratedDocument.doc_key == key))).scalars().all()
            assert gone == []
    _run(monkeypatch, body)
