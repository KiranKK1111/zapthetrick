"""Manually create the schema + run migrations against the BUNDLED docker DB.

Run from the backend venv:

    .venv\\Scripts\\python.exe run_migration.py

It forces the connection to the docker Postgres (127.0.0.1:5433, db/user/pw =
zapthetrick/postgres/zapthetrick, schema zapthetrick), then runs the same
schema-create + Alembic upgrade the app does at startup — printing each step and
the FULL traceback on any failure so we can see exactly what's wrong.
"""

from __future__ import annotations

import asyncio
import logging
import traceback

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# The bundled docker DB (matches docker-compose + routes_setup._DOCKER_PG).
DB = {
    "host": "127.0.0.1",
    "port": 5433,
    "user": "postgres",
    "password": "zapthetrick",
    "db": "postgres",          # default db; app data lives in the schema below
    "schema_name": "zapthetrick",
    "enable_age": False,       # pgvector image has no Apache AGE
}


async def main() -> None:
    from app.core.config_loader import get_config

    # Point BOTH config.yaml AND the active workspace at the bundled DB. The
    # workspace (~/.zapthetrick/workspaces.json) overrides config.yaml in the
    # storage layer, so updating config alone connects to the OLD DB.
    import app.api.routes_setup as setup

    setup._DOCKER_PG.update(DB)        # ensure the helper uses our exact target
    setup.point_app_at_docker_db()

    pg = get_config().database.postgres
    print(f"\n[target] {pg.host}:{pg.port}/{pg.db}  user={pg.user}  schema={pg.schema_name}\n")

    from storage import bootstrap as bs
    from storage.db import ensure_schema_exists, reinit_engine

    # 1) Engine on the bundled DB.
    try:
        await reinit_engine()
        print("[ok] engine -> bundled DB")
    except Exception:
        print("[FAIL] reinit_engine:")
        traceback.print_exc()
        return

    # 2) Schema + base extensions (this is what creates `zapthetrick`).
    try:
        await ensure_schema_exists()
        print("[ok] ensure_schema_exists (schema + uuid/pgcrypto/vector)")
    except Exception:
        print("[FAIL] ensure_schema_exists:")
        traceback.print_exc()

    # 3) Alembic migrations (creates the tables: sessions, messages, ...).
    try:
        bs.POSTGRES_READY = True
        from app.database import init_db

        await init_db()
        if bs.POSTGRES_READY:
            print("[ok] migrations applied")
        else:
            print(f"[FAIL] init_db left POSTGRES_READY=False: {bs.MIGRATION_ERROR}")
    except Exception:
        print("[FAIL] init_db / alembic:")
        traceback.print_exc()

    # 4) Report what's actually there now.
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=DB["host"], port=DB["port"], user=DB["user"],
            password=DB["password"], database=DB["db"],
        )
        schemas = await conn.fetch("SELECT nspname FROM pg_namespace ORDER BY 1")
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=$1 ORDER BY 1",
            DB["schema_name"],
        )
        await conn.close()
        print("\n[schemas]", [r[0] for r in schemas])
        print(f"[tables in {DB['schema_name']}]", [r[0] for r in tables])
    except Exception:
        print("[FAIL] post-check:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
