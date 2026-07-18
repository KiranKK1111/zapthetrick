"""Boot/shutdown the whole storage stack from one place.

Called from `app.main`'s FastAPI lifespan. Failures are *non-fatal*
for individual backends — if Qdrant or Dragonfly isn't reachable we
log and keep going. Postgres is the only hard dependency for routes
that read/write rows; without it the app starts in *degraded mode*
(static endpoints work; data routes 503).
"""
from __future__ import annotations

import asyncio
import logging
import socket

from .blobs.factory import close_blobs, get_blobs
from .cache.factory import close_cache, get_cache
from .db import create_engine, dispose_engine
from .graph.factory import close_graph, get_graph
from .vectors.factory import close_vector_store, get_vector_store


log = logging.getLogger(__name__)


# Module-level flag the route layer checks to short-circuit on 503
# when Postgres isn't usable (unconfigured / unreachable / mid-migrate).
POSTGRES_READY: bool = False

# Coarse state for the UI / health endpoint. Mirrors the lifecycle:
#   idle       — no config or skipped
#   migrating  — Alembic upgrade in flight (background task)
#   ready      — POSTGRES_READY True, migrations complete
#   error      — migration / connection failed; `migration_error`
#                carries the message for the UI to display
MIGRATION_STATE: str = "idle"
MIGRATION_ERROR: str | None = None


async def recheck_postgres() -> bool:
    """Live re-probe so a Postgres that came up AFTER boot self-heals.

    `POSTGRES_READY` is decided once at startup and only ever flipped OFF
    afterwards — so a DB that started late (or a transient outage that has
    since recovered) leaves the flag stale-False and the UI stuck on
    "degraded" even though the database is fully usable. This performs a
    lightweight live check and, when the DB is genuinely ready (reachable +
    a real table query succeeds), flips the flag back on so both the status
    badge and the data routes recover WITHOUT a restart. Returns the current
    readiness. No-op (returns True) when already ready.
    """
    global POSTGRES_READY, MIGRATION_STATE, MIGRATION_ERROR

    if POSTGRES_READY:
        return True
    # Never override an in-flight migration.
    if MIGRATION_STATE == "migrating":
        return False

    from app.core.config_loader import cfg
    pg = cfg.database.postgres
    if not pg.host or not pg.db or not pg.user:
        return False
    if not await _probe_tcp(pg.host, pg.port):
        return False

    try:
        from sqlalchemy import select as _select

        from storage.db import create_engine, get_session_factory
        from storage.models import Session as _SessionRow

        # Ensure the engine/session factory exists (bootstrap may have skipped
        # it in unconfigured / unreachable mode).
        if get_session_factory() is None:
            create_engine()
        factory = get_session_factory()
        if factory is None:
            return False
        # A real table query proves connectivity AND that migrations ran — a
        # bare SELECT 1 would pass on an empty, un-migrated database.
        async with factory() as s:
            await s.execute(_select(_SessionRow.id).limit(1))
    except Exception as exc:  # noqa: BLE001 — still not usable; stay degraded
        log.debug("recheck_postgres: not ready yet (%s)", exc)
        return False

    POSTGRES_READY = True
    MIGRATION_STATE = "ready"
    MIGRATION_ERROR = None
    log.info("recheck_postgres: Postgres is reachable again — recovered.")
    return True


async def _probe_tcp(host: str, port: int, *, timeout: float = 3.0) -> bool:
    """Try to open a TCP socket. Returns True if reachable, False on
    timeout / refused / unresolvable.

    Avoids asyncpg's default 60-second connect timeout — that's what
    made the uvicorn `Waiting for application startup.` log hang.
    """
    try:
        loop = asyncio.get_running_loop()

        def _connect() -> bool:
            with socket.create_connection((host, port), timeout=timeout):
                return True

        return await asyncio.wait_for(
            loop.run_in_executor(None, _connect), timeout=timeout
        )
    except Exception:
        return False


def _print_unreachable_hint(host: str, port: int) -> None:
    log.warning(
        "\n"
        "  ┌──────────────────────────────────────────────────────────┐\n"
        "  │  Postgres at %s:%d is unreachable.                       \n"
        "  │  The app is starting in DEGRADED mode — data routes will  \n"
        "  │  return 503 until the DB is up.                            \n"
        "  │                                                            \n"
        "  │  Update host / port / credentials in the app at:           \n"
        "  │     Settings -> Database -> Test connection -> Save        \n"
        "  └──────────────────────────────────────────────────────────┘\n",
        host,
        port,
    )


def _print_unconfigured_hint() -> None:
    log.warning(
        "\n"
        "  ┌──────────────────────────────────────────────────────────┐\n"
        "  │  Postgres is not configured.                              \n"
        "  │  The app started in DEGRADED mode — data routes will 503  \n"
        "  │  until you configure the database.                         \n"
        "  │                                                            \n"
        "  │  Open the app and go to:                                   \n"
        "  │     Settings -> Database                                  \n"
        "  │  Fill host / port / database / schema / user / password,   \n"
        "  │  press Test connection, then Save & migrate.               \n"
        "  └──────────────────────────────────────────────────────────┘\n",
    )


async def bootstrap_storage() -> None:
    """Build engines + warm-up clients. Idempotent.

    Three cases handled cleanly so startup never hangs:
      1. *Unconfigured* (`cfg.database.postgres.host` is empty) —
         skip everything DB-related, log a "configure in Settings"
         hint, flag `POSTGRES_READY=False`.
      2. *Unreachable* (probe times out / refuses) — same degraded
         mode, different hint pointing at host:port.
      3. *Reachable* — build the engine and let `init_db` migrate.

    Either way the app starts immediately so the Settings UI is
    available to configure / fix the database.
    """
    global POSTGRES_READY

    from app.core.config_loader import cfg

    pg = cfg.database.postgres

    # Case 1: nothing to probe — user hasn't filled in the form yet.
    if not pg.host or not pg.db or not pg.user:
        POSTGRES_READY = False
        _print_unconfigured_hint()
    else:
        reachable = await _probe_tcp(pg.host, pg.port)
        if not reachable:
            # Case 2: host/port unreachable.
            POSTGRES_READY = False
            _print_unreachable_hint(pg.host, pg.port)
        else:
            # Case 3: connection looks good. The engine call is lazy —
            # bad credentials surface from `init_db` instead.
            POSTGRES_READY = True
            create_engine()

    # Optional warm-ups. Each factory is lazy, so this just builds
    # the wrapper object without forcing a connection. Real I/O
    # happens on the first agent / route call.
    for fn, name in (
        (get_vector_store, "vector store"),
        (get_cache, "cache"),
        (get_blobs, "blobs"),
        (get_graph, "graph"),
    ):
        try:
            fn()
        except Exception as exc:
            log.warning("%s init deferred: %s", name, exc)

    # Pre-warm runs on a DELAY and only for lightweight models. Loading a big
    # embedder (e.g. bge-m3, ~2 GB) is CPU-bound and holds the GIL long enough
    # to starve the event loop — eagerly warming it at boot made the whole
    # backend unreachable. Heavy models now load lazily on first real RAG use
    # (in a worker thread), so startup stays responsive.
    asyncio.create_task(_prewarm_models(), name="prewarm-models")


# Embedders bigger than this (rough heuristic by model name) are NOT pre-warmed
# at boot — they're heavy enough that loading them, even in a worker thread,
# stalls the event loop. They load lazily on first real use instead.
_HEAVY_MODEL_HINTS = ("bge-m3", "-large", "e5-large", "gte-large", "bge-reranker-large")


def _is_heavy(model_name: str) -> bool:
    n = (model_name or "").lower()
    return any(h in n for h in _HEAVY_MODEL_HINTS)


async def _prewarm_models() -> None:
    """Best-effort cold-start of LIGHT local models, off the critical path.

    Delayed so uvicorn finishes startup and the app is reachable first. Heavy
    models are skipped entirely (they'd starve the loop on load) and warm up
    lazily on first real RAG use, inside a worker thread. Everything here is
    best-effort — failures just mean a slightly colder first request.
    """
    # Let startup settle so health/UI are responsive before we touch CPU models.
    await asyncio.sleep(20)

    try:
        from app.core.config_loader import cfg
        from app.rag import embedder

        if _is_heavy(cfg.embeddings.model):
            log.info(
                "prewarm: skipping heavy embedder %s (loads lazily on first use)",
                cfg.embeddings.model,
            )
        else:
            await asyncio.to_thread(embedder.embed_one, "warmup")
            log.info("prewarm: embedder ready (%s)", cfg.embeddings.model)
    except Exception as exc:
        log.warning("prewarm: embedder skipped (%s)", exc)

    try:
        from app.core.config_loader import cfg

        if cfg.reranker.enabled and not _is_heavy(cfg.reranker.model):
            from app.rag.retriever import _reranker  # noqa: PLC2701

            # Build + predict BOTH in a worker thread — constructing the
            # CrossEncoder on the loop was itself a stall.
            def _warm_reranker():
                r = _reranker()
                if r is not None:
                    r.predict([("warmup query", "warmup document")])

            await asyncio.to_thread(_warm_reranker)
            log.info("prewarm: reranker ready (%s)", cfg.reranker.model)
    except Exception as exc:
        log.warning("prewarm: reranker skipped (%s)", exc)


async def ensure_default_user_after_migration() -> None:
    """Create the device-local user row if it doesn't exist.

    Separate from [bootstrap_storage] because it has to run *after*
    Alembic upgrades the schema — the `users` table doesn't exist on
    a first boot until the migrations land.

    Skipped silently in degraded mode (Postgres unreachable).
    """
    if not POSTGRES_READY:
        return
    from .users import ensure_default_user

    await ensure_default_user()

    # Once the DB + user are ready, kick off a background scan that
    # re-indexes any resume whose Qdrant collection is missing.
    # Survives Qdrant volume wipes, snapshot rollbacks, and uploads
    # that landed while the vector store was unreachable — without
    # requiring the user to re-upload the PDF.
    asyncio.create_task(
        _auto_reindex_missing_resumes(), name="auto-reindex-resumes"
    )


async def _auto_reindex_missing_resumes() -> None:
    """Best-effort startup sweep that rebuilds vector indexes from
    Postgres for any resume whose collection has gone missing.

    Postgres is the source of truth (chunks live in `resume_chunks`);
    Qdrant is a rebuildable derived index. This task closes the loop
    so the architecture's actual promise — "no need to re-upload"
    — survives a Qdrant outage.

    Runs once per startup, in the background. Two correctness rules:

      1. **Every import is local** to this function. Module-load time
         is too early to touch `app.rag.*` (storage<->app cycle) and
         `embedder` lazy-loads sentence-transformers on first call.
      2. **Every `SessionFactory` lookup is late-bound** through
         `get_session_factory()`. The module attribute on `storage.db`
         is None until `create_engine()` runs in the lifespan, so a
         module-level `from .db import SessionFactory` would capture
         the original `None` and never see the post-bootstrap value
         (same bug class as the `/api/agents/stream` 'Database not
         bootstrapped' regression).
    """
    # Delay so the app is reachable BEFORE this runs. Re-embedding resumes
    # loads the embedder and is CPU-bound; doing it the instant the loop starts
    # contributed to the "backend unreachable" stall.
    await asyncio.sleep(30)

    # With a HEAVY embedder, skip the automatic boot sweep entirely — loading
    # the model + bulk re-embedding would stall the loop with no user action to
    # explain it. Resumes then rebuild lazily on demand (re-upload or POST
    # /api/resume/{id}/reindex), where the one-time load is expected.
    try:
        from app.core.config_loader import cfg as _cfg

        if _is_heavy(_cfg.embeddings.model):
            log.info(
                "auto-reindex: skipped at boot — heavy embedder %s; rebuild "
                "resume vectors on demand (re-upload or /reindex).",
                _cfg.embeddings.model,
            )
            return
    except Exception:  # noqa: BLE001 — fall through to the normal sweep
        pass

    # Local imports — keep them inside the function body. Module-load
    # time can't touch app.rag.* (circular) or sentence-transformers
    # (heavy + lazy).
    try:
        from sqlalchemy import select

        from .db import get_session_factory
        from .models import Resume
        from .vectors.factory import get_vector_store
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-reindex: import phase failed: %s", exc)
        return

    if not POSTGRES_READY:
        log.info("auto-reindex: skipped (Postgres not ready)")
        return

    factory = get_session_factory()
    if factory is None:
        log.info("auto-reindex: skipped (SessionFactory not built)")
        return

    try:
        store = get_vector_store()
        async with factory() as session:
            result = await session.execute(select(Resume))
            resumes = list(result.scalars().all())
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-reindex: resume scan failed: %s", exc)
        return

    missing: list[str] = []
    for r in resumes:
        collection = f"resume_chunks_{r.id}"
        try:
            exists = await store.has_collection(collection)
        except Exception:
            exists = False
        if not exists:
            missing.append(str(r.id))

    if not missing:
        log.info("auto-reindex: all %d resume(s) have vectors", len(resumes))
        return

    log.warning(
        "auto-reindex: %d resume(s) missing vectors; rebuilding from Postgres",
        len(missing),
    )

    # Late-import the rebuild path so a sentence-transformers absence
    # only matters if there's actually something to reindex.
    try:
        from app.rag import embedder
        from app.rag import store as vec_store
        from app.rag.retriever import invalidate_bm25
        from .repos import ResumeRepo
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "auto-reindex: rebuild deps unavailable (%s); leaving %d resume(s) "
            "without vectors. Use POST /api/resume/{id}/reindex manually.",
            exc,
            len(missing),
        )
        return

    for resume_id in missing:
        try:
            # Re-acquire the factory each iteration — a slow reindex
            # can outlive an engine swap from a Settings save, and the
            # late-bound lookup picks up the new one transparently.
            cur_factory = get_session_factory()
            if cur_factory is None:
                log.warning(
                    "auto-reindex: engine gone mid-loop; aborting after %d",
                    missing.index(resume_id),
                )
                return
            async with cur_factory() as session:
                repo = ResumeRepo(session)
                chunks = await repo.fetch_chunks(resume_id)
                if not chunks:
                    continue
                invalidate_bm25(resume_id)
                texts = [c.content for c in chunks]
                embeddings = await asyncio.to_thread(embedder.embed, texts)
                await vec_store.upsert(
                    ids=[str(c.vector_point_id or c.id) for c in chunks],
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "resume_id": resume_id,
                            "chunk_id": str(c.id),
                            "position": c.position,
                            "section": c.section_type or "",
                        }
                        for c in chunks
                    ],
                )
            log.info(
                "auto-reindex: rebuilt %d chunks for resume %s",
                len(chunks),
                resume_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "auto-reindex: skipped resume %s — %s", resume_id, exc
            )


async def run_migrations_in_background() -> None:
    """Apply Alembic migrations off the request path.

    Wraps [app.database.init_db] so the lifespan can fire it as a
    task and yield immediately — uvicorn finishes startup, the app
    accepts requests, and any data route 503s with a clear message
    until [MIGRATION_STATE] flips to `ready`.

    Idempotent and exception-safe; logs the state transitions.
    """
    global MIGRATION_STATE, MIGRATION_ERROR, POSTGRES_READY

    # The probe in `bootstrap_storage` already decided whether
    # there's anything to do. Skip if we're in unconfigured /
    # unreachable mode.
    if not POSTGRES_READY:
        MIGRATION_STATE = "idle"
        MIGRATION_ERROR = None
        log.info("migration: skipped (POSTGRES_READY is False)")
        return

    MIGRATION_STATE = "migrating"
    MIGRATION_ERROR = None
    log.info("migration: starting")
    try:
        # Lazy import — `app.database` depends on the engine being built,
        # which `bootstrap_storage` already did.
        from app.database import init_db

        await init_db()
        # init_db flips POSTGRES_READY back to False on failure.
        if POSTGRES_READY:
            await ensure_default_user_after_migration()
            MIGRATION_STATE = "ready"
            log.info("migration: complete; data routes ready")
        else:
            MIGRATION_STATE = "error"
            MIGRATION_ERROR = MIGRATION_ERROR or "init_db marked degraded"
            log.error("migration: init_db left POSTGRES_READY=False")
    except Exception as exc:  # noqa: BLE001
        POSTGRES_READY = False
        MIGRATION_STATE = "error"
        MIGRATION_ERROR = str(exc)
        log.exception("migration: failed (%s)", exc)


async def shutdown_storage() -> None:
    """Tear everything down. Best-effort; failures are logged + swallowed."""
    for closer, name in (
        (close_graph, "graph"),
        (close_blobs, "blobs"),
        (close_cache, "cache"),
        (close_vector_store, "vectors"),
        (dispose_engine, "postgres"),
    ):
        try:
            await closer()
        except Exception as exc:
            log.warning("shutdown %s failed: %s", name, exc)
