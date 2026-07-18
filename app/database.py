"""Legacy `app.database` — now a thin shim over `storage.*`.

The real source of truth is [app.storage.models] for ORM and
[app.storage.db] for the engine + session factory. Everything in this
file is a re-export so old imports keep working:

    from app.database import Message, Conversation, get_session   # still works

Module-level callable shims:
  - `init_db()`   — applies Alembic migrations (or a no-op if disabled
                    via `cfg.database.migrations.auto_apply`).
  - `get_session()` — yields a Postgres-backed async session.
  - `SessionFactory` — late-bound to the factory in [storage.db].

Field-rename compat for the v1 schema is handled at the ORM level via
`synonym()` (e.g. `Message.conversation_id` is a synonym for
`session_id`). See [storage.models].
"""
from __future__ import annotations

from storage import db as _db
from storage.models import (
    AgentRun,
    Base,
    Episode,
    Feedback,
    Message,
    ModelUsage,
    Project,
    Resume,
    ResumeChunk,
    Session,
    SkillRow,
    User,
)

# v1 names → v2 models.
Conversation = Session

__all__ = [
    "Base",
    "Conversation",
    "Project",
    "Session",
    "Message",
    "Resume",
    "ResumeChunk",
    "Episode",
    "SkillRow",
    "Feedback",
    "AgentRun",
    "ModelUsage",
    "User",
    "SessionFactory",
    "get_session",
    "init_db",
]


def __getattr__(name: str):
    """Late-bind `SessionFactory` so callers see the current value even
    if they `from app.database import SessionFactory` at import time."""
    if name == "SessionFactory":
        return _db.SessionFactory
    raise AttributeError(name)


async def get_session():
    """Yield a Postgres-backed AsyncSession. Same API as before."""
    async for session in _db.get_session():
        yield session


async def init_db() -> None:
    """Boot the database — applies Alembic migrations if auto-apply is on.

    Schema creation moved to Alembic per DataBaseArchitecture.md. We
    keep this function so the existing `app.main` lifespan call site
    doesn't need to change.

    Skipped silently when Postgres is unreachable — the bootstrap
    layer already logged the docker hint and flipped `POSTGRES_READY`
    to False. Running Alembic here would just hang on asyncpg's
    connect timeout.
    """
    from app.core.config_loader import cfg
    from storage.bootstrap import POSTGRES_READY

    if not POSTGRES_READY:
        return

    # Auto-create the configured schema if it doesn't exist. Lets the
    # Settings UI's "Schema" field be fire-and-forget — type a name,
    # hit Save, schema appears, migrations run inside.
    try:
        await _db.ensure_schema_exists()
    except Exception as exc:
        import logging

        from storage import bootstrap as _bs

        logging.getLogger(__name__).error(
            "CREATE SCHEMA IF NOT EXISTS failed: %s — degraded mode.", exc
        )
        _bs.POSTGRES_READY = False
        return

    # Make sure the engine + SessionFactory are built first.
    _db.create_engine()

    if not cfg.database.migrations.auto_apply:
        return

    # Run `alembic upgrade head` in-process.
    import os

    from alembic import command
    from alembic.config import Config

    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cfg_path = os.path.join(backend_root, "alembic.ini")
    if not os.path.exists(cfg_path):
        # Dev install without alembic.ini? Don't crash startup.
        return
    alembic_cfg = Config(cfg_path)
    # Pin script_location to an ABSOLUTE path. alembic.ini's relative
    # `storage/migrations` resolves against the cwd, which is wrong in a
    # PyInstaller-frozen app (the migration scripts live in the bundle dir,
    # i.e. backend_root here). Absolute makes dev + frozen both work.
    alembic_cfg.set_main_option(
        "script_location", os.path.join(backend_root, "storage", "migrations")
    )
    # Build the URL the same way storage/db.py does.
    pg = cfg.database.postgres
    alembic_cfg.set_main_option(
        "sqlalchemy.url",
        f"postgresql+asyncpg://{pg.user}:{pg.password}"
        f"@{pg.host}:{pg.port}/{pg.db}",
    )
    # Alembic's `command.upgrade` is synchronous; offload it. Catch
    # connection / auth errors here so a bad password doesn't crash
    # startup — the route layer will surface a clear 503 on first
    # request and `POSTGRES_READY` is set to False below.
    import asyncio
    import logging

    log = logging.getLogger(__name__)
    try:
        await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
    except Exception as exc:
        log.error(
            "\n"
            "  ┌──────────────────────────────────────────────────────────┐\n"
            "  │  Alembic migration failed: %s\n"
            "  │  Postgres is reachable at %s:%d but rejected the          \n"
            "  │  connection. Check `cfg.database.postgres.{user,password,db}`\n"
            "  │  in config.yaml. If you're running the bundled            \n"
            "  │  docker-compose, the password is `local`.                  \n"
            "  └──────────────────────────────────────────────────────────┘",
            exc,
            pg.host,
            pg.port,
        )
        # Mark as degraded so data routes return 503 instead of
        # hanging on every retry.
        from storage import bootstrap as _bs

        _bs.POSTGRES_READY = False
