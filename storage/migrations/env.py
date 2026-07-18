"""Alembic env — async-aware, autogen-aware, schema-aware.

Pulls the DB URL at runtime from `cfg.database.postgres` so changing
the host (or schema) in config.yaml is enough — no `alembic.ini`
edits. The schema is applied via asyncpg's `server_settings.search_path`
so new tables land where `cfg.database.postgres.schema_name` says.
"""
from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config


# `prepend_sys_path = .` in alembic.ini already adds `backend/`; this
# is a safety net for `python -m alembic`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config_loader import cfg  # noqa: E402
from storage.models import Base  # noqa: E402


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _runtime_url() -> str:
    pg = cfg.database.postgres
    return (
        f"postgresql+asyncpg://{pg.user}:{pg.password}"
        f"@{pg.host}:{pg.port}/{pg.db}"
    )


def _schema_name() -> str:
    """Active schema. Always coerce empty/None back to `public`."""
    return (cfg.database.postgres.schema_name or "public").strip() or "public"


def _search_path() -> str:
    """Schema search_path applied to each migration connection.

    Mirrors `storage.db._search_path` — `public` is always appended so
    shared extensions (uuid-ossp, pgcrypto, AGE) resolve cleanly.
    """
    schema = _schema_name()
    return "public" if schema == "public" else f"{schema},public"


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Render SQL to stdout — useful for review without a live DB."""
    context.configure(
        url=_runtime_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=_schema_name(),
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Pin the version table to the configured schema so `alembic_version`
    # lives with the rest of the app's tables, not in public.
    schema = _schema_name()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=schema,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    cfg_section = config.get_section(config.config_ini_section, {})
    cfg_section["sqlalchemy.url"] = _runtime_url()
    connectable = async_engine_from_config(
        cfg_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={
            "server_settings": {"search_path": _search_path()},
        },
    )
    async with connectable.connect() as conn:
        # Make sure the schema exists before CREATE TABLE runs. Idempotent.
        schema = _schema_name()
        if schema != "public":
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            await conn.commit()
        await conn.run_sync(do_run_migrations)
        # Commit the migration work. SQLAlchemy 2.0 async connections ROLL
        # BACK any open transaction when the `async with` block exits, and
        # alembic's begin_transaction() inside run_sync doesn't own the outer
        # async transaction — without this commit every table created above
        # silently vanished on close (observed in the cluster: schema
        # persisted, zero tables, migrations replayed on every boot). No-op
        # when alembic already committed everything itself.
        await conn.commit()
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
