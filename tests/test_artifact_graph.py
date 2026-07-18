"""Phase 4 #10/#11/#17 — artifact dependency graph + stale-marking, incremental
patch (wires apply_patch), and universal undo.
"""
from __future__ import annotations

import asyncio

from app.artifacts.store import ArtifactStore


def _store():
    return ArtifactStore()


# ── #10 dependency graph + stale-marking ────────────────────────────────────
def test_dependent_marked_stale_on_upstream_change():
    async def go():
        s = _store()
        src = await s.create("ws", "code", "src", "print(1)", "py")
        doc = await s.create("ws", "document", "doc", "# Docs", "md")
        s.add_dependency(doc.id, src.id)          # doc derived from src
        assert not s.is_stale(doc.id)
        # upstream changes → dependent goes stale
        await s.append_version(src.id, "print(2)")
        assert s.is_stale(doc.id)
        assert doc.id in s.stale_dependents(src.id)
        # regenerating the dependent clears its stale flag
        await s.append_version(doc.id, "# Docs v2")
        assert not s.is_stale(doc.id)
    asyncio.run(go())


def test_transitive_stale_marking():
    async def go():
        s = _store()
        a = await s.create("ws", "code", "a", "a", "py")
        b = await s.create("ws", "code", "b", "b", "py")
        c = await s.create("ws", "code", "c", "c", "py")
        s.add_dependency(b.id, a.id)              # b <- a
        s.add_dependency(c.id, b.id)              # c <- b
        await s.append_version(a.id, "a2")
        assert s.is_stale(b.id) and s.is_stale(c.id)   # transitive
    asyncio.run(go())


# ── #11 incremental patch (wires artifacts/patch.apply_patch) ────────────────
def test_patch_version_applies_and_appends():
    async def go():
        s = _store()
        art = await s.create("ws", "document", "d", "Use MySQL here.", "md")
        ver, applied = await s.patch_version(art.id, "replace MySQL with PostgreSQL")
        assert applied and ver is not None
        assert art.current_version == 2
        content = (await s.content(art.id)).decode()
        assert "PostgreSQL" in content and "MySQL" not in content
    asyncio.run(go())


def test_patch_version_unappliable_falls_back():
    async def go():
        s = _store()
        art = await s.create("ws", "document", "d", "nothing here", "md")
        ver, applied = await s.patch_version(art.id, "replace Redis with Memcached")
        assert not applied and ver is None
        assert art.current_version == 1          # no new version created
    asyncio.run(go())


# ── #17 universal undo ──────────────────────────────────────────────────────
def test_undo_restores_previous_version():
    async def go():
        s = _store()
        art = await s.create("ws", "document", "d", "v1", "md")
        await s.append_version(art.id, "v2")
        assert (await s.content(art.id)).decode() == "v2"
        ver = await s.undo(art.id)
        assert ver is not None
        # undo appended v1's bytes as a new (v3) version, non-destructively
        assert (await s.content(art.id)).decode() == "v1"
        assert art.current_version == 3
    asyncio.run(go())


def test_undo_with_no_history_is_none():
    async def go():
        s = _store()
        art = await s.create("ws", "document", "d", "only", "md")
        assert await s.undo(art.id) is None
    asyncio.run(go())
