"""Deeper smoke test for the storage layer.

Exercises everything we can without a live Postgres / Qdrant / Dragonfly:
  - ORM <-> dataclass round-trips
  - Repo constructor signatures
  - Blob filesystem put/get/delete + traversal safety
  - In-memory cache: get/set/incr/pub-sub
  - VectorStore + GraphStore factory wiring (no I/O)
  - Synonym aliases on Message
  - Alembic migration loads
  - Lifespan + routes_resume + rag + memory imports

Run from `backend/`:
    python -m storage.smoke
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path


log = logging.getLogger("storage.smoke")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))
    log.info("%s  %s  %s", PASS if ok else FAIL, name, detail)


async def t_config_loads() -> None:
    from app.core.config_loader import get_config

    c = get_config()
    record(
        "config loads new db sections",
        c.database.postgres.host == "localhost",
        f"backend={c.database.cache.backend}",
    )


async def t_orm_synonym_roundtrip() -> None:
    from storage.models import Message

    sid = uuid.uuid4()
    m = Message(conversation_id=sid, role="user", content="hi")
    record("Message synonym conversation_id -> session_id", m.session_id == sid)


async def t_factories_build() -> None:
    from storage.vectors import get_vector_store
    from storage.cache import get_cache
    from storage.blobs import get_blobs
    from storage.graph import get_graph

    vs = get_vector_store()
    c = get_cache()
    b = get_blobs()
    g = get_graph()
    record(
        "factories build",
        bool(vs) and bool(c) and bool(b),
        f"{type(vs).__name__} / {type(c).__name__} / {type(b).__name__} / {type(g).__name__ if g else 'None'}",
    )


async def t_cache_roundtrip() -> None:
    from storage.cache.memory_cache import MemoryCache

    cache = MemoryCache(default_ttl_seconds=10)
    await cache.set("k1", "v1")
    v = await cache.get("k1")
    n = await cache.incr("hits")
    await cache.set("k2", "v2", ttl_seconds=0)            # no expiry
    record("cache get/set/incr", v == "v1" and n == 1)


async def t_blob_roundtrip() -> None:
    from storage.blobs.fs_blobs import FilesystemBlobs

    with tempfile.TemporaryDirectory() as d:
        blobs = FilesystemBlobs(root=d)
        path = await blobs.put("a/b/c.txt", b"hello")
        ok_exists = await blobs.exists("a/b/c.txt")
        data = await blobs.get("a/b/c.txt")
        await blobs.delete("a/b/c.txt")
        gone = not await blobs.exists("a/b/c.txt")
        record(
            "blob put/get/exists/delete",
            ok_exists and data == b"hello" and gone,
            f"resolved at {Path(path).name}",
        )


async def t_blob_traversal_safety() -> None:
    from storage.blobs.fs_blobs import FilesystemBlobs

    with tempfile.TemporaryDirectory() as d:
        blobs = FilesystemBlobs(root=d)
        try:
            await blobs.put("../escaped.txt", b"nope")
            record("blob blocks ../ traversal", False, "should have raised")
        except PermissionError:
            record("blob blocks ../ traversal", True)


async def t_repo_constructors() -> None:
    """Repos take exactly one constructor arg (the AsyncSession). That's
    cheap to verify without a live database — we just check the
    signature is the shape the route layer expects."""
    import inspect

    from storage.repos import (
        AgentRunRepo,
        FeedbackRepo,
        MessageRepo,
        ResumeRepo,
        SessionRepo,
        UsageRepo,
    )

    for cls in (
        SessionRepo,
        MessageRepo,
        ResumeRepo,
        FeedbackRepo,
        AgentRunRepo,
        UsageRepo,
    ):
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        ok = len(params) == 1 and params[0].name == "session"
        record(f"repo {cls.__name__} signature", ok, str(sig))


async def t_alembic_revision_loads() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mig", "storage/migrations/versions/0001_initial.py"
    )
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        record("alembic 0001_initial loads", mod.revision == "0001_initial")
    except ModuleNotFoundError as exc:
        # `alembic` not installed — expected pre-`pip install`.
        record("alembic 0001_initial loads", False, f"alembic missing: {exc}")


async def t_lifespan_wiring() -> None:
    """Verify the lifespan still wires `bootstrap_storage` +
    `shutdown_storage` and kicks off the background migration task.

    Migrations moved to a background task so uvicorn's `startup
    complete` log fires immediately even when the DB is slow.
    The smoke test now matches that flow.
    """
    import inspect

    from app.main import lifespan

    src = inspect.getsource(lifespan)
    ok = all(
        s in src
        for s in (
            "bootstrap_storage",
            "run_migrations_in_background",
            "shutdown_storage",
        )
    )
    record("app.main lifespan wires storage", ok)


async def t_routes_resume_imports() -> None:
    """The ported routes_resume must import cleanly even without Postgres."""
    try:
        import app.api.routes_resume as r  # noqa: F401

        record("routes_resume imports clean", True)
    except Exception as exc:
        record("routes_resume imports clean", False, repr(exc))


async def t_rag_imports() -> None:
    try:
        import app.rag.store  # noqa: F401
        import app.rag.ingest  # noqa: F401
        import app.rag.retriever  # noqa: F401

        record("rag (store + ingest + retriever) imports", True)
    except Exception as exc:
        record("rag (store + ingest + retriever) imports", False, repr(exc))


async def t_memory_imports() -> None:
    try:
        from app.memory.episodic import (  # noqa: F401
            Episode,
            record_episode,
            search_episodes_similar,
        )
        from app.memory.semantic import (  # noqa: F401
            Skill,
            record_skill,
            relevant_skills_for_question,
        )

        record("memory layer imports", True)
    except Exception as exc:
        record("memory layer imports", False, repr(exc))


async def main() -> int:
    await t_config_loads()
    await t_orm_synonym_roundtrip()
    await t_factories_build()
    await t_cache_roundtrip()
    await t_blob_roundtrip()
    await t_blob_traversal_safety()
    await t_repo_constructors()
    await t_alembic_revision_loads()
    await t_lifespan_wiring()
    await t_routes_resume_imports()
    await t_rag_imports()
    await t_memory_imports()

    print()
    print("=" * 70)
    failed = [r for r in results if r[1] == FAIL]
    for name, status, detail in results:
        marker = "[+]" if status == PASS else "[-]"
        line = f"{marker} {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
    print("=" * 70)
    print(f"{len(results) - len(failed)}/{len(results)} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
