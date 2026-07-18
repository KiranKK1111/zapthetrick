"""Phase 4 #20/#21 — universal doc & code transformation wired from the artifact
side, plus the resume-template transformer.
"""
from __future__ import annotations

import asyncio

from app.artifacts import transform_bridge as tb
from app.artifacts.store import ArtifactStore


def test_run_transform_formats_markdown():
    async def go():
        res = await tb.run_transform("# Title\n\nsome text", filename="notes.md")
        # documents layer available in-repo → a real TransformResult
        assert res is not None
        assert res.kind == "markdown"
        assert res.content
    asyncio.run(go())


def test_transform_with_injected_transformer_runs_it():
    async def go():
        async def up(x):
            return x.upper()
        res = await tb.run_transform("hello", filename="a.md", transformer=up)
        assert res is not None
        assert "HELLO" in res.content
    asyncio.run(go())


def test_transform_and_store_persists_version():
    async def go():
        store = ArtifactStore()
        art, res = await tb.transform_and_store(
            store, "ws", "# Doc\n\nbody", title="My Doc", filename="d.md")
        assert art is not None
        assert store.get(art.id) is not None
        assert art.current_version == 1
    asyncio.run(go())


def test_resume_transformer_is_callable():
    async def go():
        tx = tb.resume_transformer("classic")
        out = await tx("# Experience\n\n- did things")
        assert isinstance(out, str) and out
    asyncio.run(go())
