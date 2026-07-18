"""Artifact store + creation discipline (workspace-and-artifacts R4/R5, task 4.3).

Pins Properties 4 & 5: discipline (substantial structured output → artifact;
normal answer → none), create/append/version/restore, bounded retention, and
blob persistence via the injected backend.
"""
from __future__ import annotations

import asyncio

from app.artifacts.discipline import should_create_artifact
from app.artifacts.store import ArtifactStore


# ── creation discipline (Property 4) ─────────────────────────────────────────
def test_normal_answer_makes_no_artifact():
    assert should_create_artifact("Sure, here's a quick tip.") is None
    assert should_create_artifact("Yes.") is None


def test_explicit_document_request_makes_artifact():
    assert should_create_artifact("...", explicit_format="pdf") == "document"


def test_doc_flagged_sources_makes_artifact():
    assert should_create_artifact("long answer", sources={"document": True}) == "document"


def test_mermaid_is_diagram():
    assert should_create_artifact("```mermaid\ngraph TD; A-->B\n```") == "diagram"


def test_substantial_code_is_code_artifact():
    body = "Here is the module:\n```python\n" + ("x = 1\n" * 100) + "```"
    assert should_create_artifact(body, min_chars=50) == "code"


def test_sql_block_is_sql_artifact():
    body = "```sql\nCREATE TABLE users (id int);\n```" + ("\n-- note" * 40)
    assert should_create_artifact(body, min_chars=50) == "sql"


def test_long_markdown_is_markdown_artifact():
    body = "# Title\n\n" + ("Lots of structured prose. " * 40) + "\n## Section\n"
    assert should_create_artifact(body, min_chars=50) == "markdown"


# ── store + versioning (Property 5) ──────────────────────────────────────────
def test_create_and_append_versions_monotonic():
    s = ArtifactStore()

    async def run():
        art = await s.create("ws1", "markdown", "Doc", "# v1", "md")
        await s.append_version(art.id, "# v1\n## added", "md")
        return art

    art = asyncio.run(run())
    vers = s.versions(art.id)
    assert [v.version for v in vers] == [1, 2]
    assert s.get(art.id).current_version == 2


def test_per_version_content_retrieval():
    # The inter-version diff + editable-pane endpoints read a specific version's
    # bytes via store.content(id, version) — each version returns its own text.
    s = ArtifactStore()

    async def run():
        art = await s.create("ws1", "code", "Snippet", "print(1)", "py")
        await s.append_version(art.id, "print(1)\nprint(2)", "py")
        v1 = await s.content(art.id, 1)
        v2 = await s.content(art.id, 2)
        cur = await s.content(art.id)  # None → current
        return v1, v2, cur

    v1, v2, cur = asyncio.run(run())
    assert v1 == b"print(1)"
    assert v2 == b"print(1)\nprint(2)"
    assert cur == v2


def test_restore_appends_old_content_as_new_version():
    s = ArtifactStore()

    async def run():
        art = await s.create("ws1", "markdown", "Doc", "original", "md")
        await s.append_version(art.id, "edited", "md")
        restored = await s.restore(art.id, 1)
        body = await s.content(art.id)        # current = restored
        return art, restored, body

    art, restored, body = asyncio.run(run())
    assert restored.version == 3              # new monotonic version
    assert body == b"original"                # content matches v1


def test_retention_bound_evicts_oldest():
    s = ArtifactStore(max_versions=3)

    async def run():
        art = await s.create("ws1", "markdown", "Doc", "v1", "md")
        for i in range(2, 7):
            await s.append_version(art.id, f"v{i}", "md")
        return art

    art = asyncio.run(run())
    vers = s.versions(art.id)
    assert len(vers) == 3
    # Oldest (v1..v3) evicted; the newest three remain, numbers still monotonic.
    assert [v.version for v in vers] == [4, 5, 6]


def test_blob_backend_is_used_when_injected():
    store_bytes: dict = {}

    async def put(ref, data):
        store_bytes[ref] = data
        return ref

    async def get(ref):
        return store_bytes[ref]

    s = ArtifactStore(blob_put=put, blob_get=get)

    async def run():
        art = await s.create("ws1", "markdown", "Doc", "hello", "md")
        return await s.content(art.id)

    body = asyncio.run(run())
    assert body == b"hello"
    assert store_bytes                          # bytes went to the injected blob
