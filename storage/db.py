"""Async Postgres engine + session factory.

Lazy, idempotent, and reachable from any module. Boot sequence:

  bootstrap (app.main lifespan)
    -> create_engine(url)              # builds the async engine
    -> SessionFactory() ...            # used everywhere downstream

The engine is held at module level so request handlers and background
agents share one pool. `pool_pre_ping=True` keeps connections healthy
across Postgres restarts (common in dev with `docker compose restart`).

Schema (`cfg.database.postgres.schema_name`) is applied via asyncpg's
`server_settings.search_path` so every connection out of the pool
defaults to that schema. The shim helper [ensure_schema_exists] runs
`CREATE SCHEMA IF NOT EXISTS` against the maintenance database before
Alembic touches anything.

The URL is built from `cfg.database.postgres.*`. We deliberately don't
parse it from a single string — the spec wants each piece configurable
through the settings UI.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config_loader import cfg


log = logging.getLogger(__name__)


_engine: AsyncEngine | None = None
SessionFactory: async_sessionmaker[AsyncSession] | None = None  # set in [create_engine]


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Late-bound accessor for the global [SessionFactory].

    Migrations now run as a background task, so route modules that did
    `from app.database import SessionFactory` at import time captured
    `None` and never saw the post-bootstrap value. Use this helper
    instead — each call returns the *current* SessionFactory, or None
    if Postgres isn't usable yet.
    """
    return SessionFactory


def _apply_active_workspace_to_cfg() -> None:
    """Project the active workspace's `relational` slot into
    `cfg.database.postgres.*` so the engine builder + Alembic see the
    right values.

    Architecture.md §"Workspace": activating a workspace in the UI
    must switch the live DB. The workspace repo writes to
    `~/.zapthetrick/workspaces.json`; this helper reads from there
    and overrides cfg.

    No-op when:
      - no workspace is active (fresh install before the user runs
        the setup screen)
      - the active workspace doesn't declare a relational driver
      - the driver isn't one we know how to map to cfg.database.postgres

    Today we only support the postgres + sqlite drivers — sqlite
    routes through a sqlite+aiosqlite URL transparently. MySQL is
    stubbed: the workspace probe lets the user verify reachability
    but the engine still needs an asyncpg+postgres or sqlite
    connection here.
    """
    try:
        from app.workspace import default_workspace_repo
    except Exception:  # noqa: BLE001 — workspace module is optional pre-bootstrap
        return
    repo = default_workspace_repo()
    active = repo.active()
    if active is None:
        return
    relational = active.relational or {}
    driver = relational.get("driver")
    if driver == "postgres":
        pg = cfg.database.postgres
        # Only override fields actually present — empty fields fall
        # through to the cfg defaults (which the YAML / Settings UI
        # set). Passwords masked as ******** are preserved as-is
        # because the workspace UI never overwrites them with the mask.
        for key, attr in (
            ("host", "host"),
            ("port", "port"),
            ("db", "db"),
            ("schema_name", "schema_name"),
            ("user", "user"),
            ("password", "password"),
        ):
            v = relational.get(key)
            if v not in (None, "", "********"):
                setattr(pg, attr, v)
        log.info(
            "workspace -> cfg.database.postgres: host=%s db=%s schema=%s",
            pg.host, pg.db, pg.schema_name,
        )

    # Vector store slot — vectors live in Postgres via pgvector, so there is no
    # separate vector-DB connection to route from the workspace UI.

    # Cache slot.
    cache = active.cache or {}
    cdriver = cache.get("driver")
    if cdriver == "redis":
        c = cfg.database.cache
        for key, attr in (("url", "url"), ("default_ttl_seconds", "default_ttl_seconds")):
            v = cache.get(key)
            if v not in (None, "", "********"):
                setattr(c, attr, v)
        log.info("workspace -> cfg.database.cache: url=%s", c.url)

    # Blob slot.
    blob = active.blob or {}
    bdriver = blob.get("driver")
    if bdriver == "filesystem":
        s = cfg.database.storage
        v = blob.get("path")
        if v not in (None, "", "********"):
            s.blobs_path = v
        log.info("workspace -> cfg.database.storage: blobs_path=%s", s.blobs_path)


def _build_url() -> str:
    # Refresh cfg.database.postgres from the active workspace before
    # composing the URL. Cheap (file read) and idempotent.
    _apply_active_workspace_to_cfg()
    pg = cfg.database.postgres
    return (
        f"postgresql+asyncpg://{pg.user}:{pg.password}"
        f"@{pg.host}:{pg.port}/{pg.db}"
    )


def _search_path() -> str:
    """Build the search_path string applied per-connection.

    Always includes `public` after the user's schema so shared
    extensions (uuid-ossp, pgcrypto, AGE) resolve.
    """
    schema = (cfg.database.postgres.schema_name or "public").strip()
    if schema == "public":
        return "public"
    return f"{schema},public"


def create_engine() -> AsyncEngine:
    """Build (or return the cached) async engine.

    Safe to call repeatedly — successive calls return the same engine.
    Pool sizing comes from `cfg.database.postgres.pool_min / pool_max`.
    """
    global _engine, SessionFactory
    if _engine is not None:
        return _engine
    pg = cfg.database.postgres
    _engine = create_async_engine(
        _build_url(),
        pool_size=pg.pool_min,
        max_overflow=max(0, pg.pool_max - pg.pool_min),
        pool_pre_ping=True,
        echo=False,
        future=True,
        # asyncpg honours `server_settings` for per-session GUCs. The
        # search_path is what makes `cfg.database.postgres.schema_name`
        # actually take effect — every query without an explicit schema
        # qualifier resolves there first.
        connect_args={
            "server_settings": {"search_path": _search_path()},
        },
    )
    SessionFactory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    """Tear down the pool. Called from the FastAPI lifespan on shutdown."""
    global _engine, SessionFactory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    SessionFactory = None


async def reinit_engine() -> None:
    """Dispose + rebuild — used after the user edits DB settings in the UI.

    The new pool picks up host/port/db/schema/user/password from
    `cfg.database.postgres.*` on the next `create_engine` call.

    Also resets the in-memory device-user cache: if the user docked a
    different database the cached UUID is from the *previous* DB and
    every `WHERE user_id = <stale-uuid>` query would silently return
    zero rows — which is exactly the symptom users see when
    "the uploaded resume is gone after re-docking the database".
    """
    await dispose_engine()
    create_engine()
    try:
        from .device import reset_cache_for_tests as _reset_device_cache

        _reset_device_cache()
    except Exception:  # noqa: BLE001 — never fail a reinit on cache hygiene
        pass


async def ensure_schema_exists() -> None:
    """Prepare the configured database for migrations.

    Three things happen here, all idempotent:

      1. `CREATE EXTENSION IF NOT EXISTS "uuid-ossp"` + `pgcrypto` —
         required by the Alembic migration's `uuid_generate_v4()`
         defaults and any future PII column encryption. Both are
         standard, ship with every Postgres install; the old
         docker-compose seeded them via init scripts, so for a local
         Postgres we do it here.

      2. Apache AGE — optional. We try `CREATE EXTENSION age` and
         create the `kg` graph; silently skipped when the binary
         isn't installed (typical for vanilla Postgres). The graph
         factory then returns None and the rest of the app keeps
         working.

      3. `CREATE SCHEMA IF NOT EXISTS` for the user's configured
         schema.

    Uses asyncpg directly (no engine) so this works *before* the
    pool is built. The configured user needs CREATE EXTENSION /
    CREATE SCHEMA privileges; superuser has both. Extension failures
    are logged but don't abort — the Alembic migration would surface
    a more actionable error than this layer can.
    """
    pg = cfg.database.postgres
    schema = (pg.schema_name or "public").strip()
    conn = None
    try:
        log.info("ensure_schema_exists: connecting to %s:%d/%s ...", pg.host, pg.port, pg.db)
        conn = await asyncpg.connect(
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.db,
            timeout=5.0,
        )
        log.info("ensure_schema_exists: connected")

        # Create the schema(s) BEFORE anything else and pin the search_path.
        # `CREATE SCHEMA` doesn't depend on search_path, so this works even when
        # the connection's default search_path is empty or points at a schema
        # that doesn't exist yet — which is exactly what causes
        # "no schema has been selected to create in" on the CREATE EXTENSION
        # calls below. With the schema created + search_path pinned, the
        # extensions have a valid target.
        await conn.execute('CREATE SCHEMA IF NOT EXISTS public')
        if schema and schema != "public":
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            await conn.execute(f'SET search_path TO "{schema}", public')
        else:
            await conn.execute("SET search_path TO public")
        log.info("ensure_schema_exists: schema %r ready", schema)

        # Standard extensions — needed by the initial migration.
        for ext in ("uuid-ossp", "pgcrypto"):
            try:
                await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                log.info("ensure_schema_exists: extension %s ready", ext)
            except Exception as exc:
                log.warning(
                    "CREATE EXTENSION %s failed: %s — migrations may fail.",
                    ext,
                    exc,
                )

        # Apache AGE — optional / non-standard. Bound the call with a
        # short timeout because some half-installed AGE setups make
        # CREATE EXTENSION hang indefinitely (waits on shared-lib load).
        if pg.enable_age:
            try:
                import asyncio as _asyncio

                async def _try_age():
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS age')
                    await conn.execute("LOAD 'age'")
                    already = await conn.fetchval(
                        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'kg'"
                    )
                    if not already:
                        await conn.execute("SELECT create_graph('kg')")

                await _asyncio.wait_for(_try_age(), timeout=3.0)
                log.info("Apache AGE graph 'kg' ready")
            except Exception as exc:
                log.info(
                    "Apache AGE not available (%s) — graph features disabled.",
                    exc,
                )

        # (Schema + search_path were established at the top, before the
        # extensions, so there's nothing left to do here.)
    finally:
        if conn is not None:
            await conn.close()


async def test_connection(*, postgres_overrides: dict | None = None) -> dict:
    """Try a one-shot connection with optional `postgres_overrides`.

    Used by `POST /api/settings/database/test` so the user can verify
    new credentials before clicking Save. Returns a structured result
    rather than raising — the UI shows green/red without parsing
    stacktraces.
    """
    pg = cfg.database.postgres
    overrides = postgres_overrides or {}
    host = overrides.get("host", pg.host)
    port = int(overrides.get("port", pg.port))
    db = overrides.get("db", pg.db)
    user = overrides.get("user", pg.user)
    password = overrides.get("password", pg.password)
    schema = overrides.get("schema_name", pg.schema_name) or "public"

    conn = None
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db,
            timeout=5.0,
        )
        version = await conn.fetchval("SELECT version()")
        # Verify we can list (or implicitly create) the schema.
        has_schema = await conn.fetchval(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
            schema,
        )
        return {
            "ok": True,
            "version": version,
            "schema_exists": bool(has_schema),
            "schema": schema,
            "db": db,
            "host": host,
            "port": port,
        }
    except asyncpg.InvalidPasswordError:
        return {"ok": False, "error": "Authentication failed (bad user/password).", "host": host, "port": port}
    except asyncpg.InvalidCatalogNameError:
        return {"ok": False, "error": f"Database {db!r} does not exist.", "host": host, "port": port}
    except OSError as exc:
        return {"ok": False, "error": f"Cannot reach {host}:{port} — {exc}", "host": host, "port": port}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "host": host, "port": port}
    finally:
        if conn is not None:
            await conn.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a session and commits/rolls back."""
    if SessionFactory is None:
        create_engine()
    assert SessionFactory is not None
    async with SessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
