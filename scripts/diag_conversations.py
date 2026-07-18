"""Diagnostic: query `sessions` + `messages` directly and print what the
chat-history endpoints would return.

Mirrors the logic in `routes_chat.list_conversations` /
`get_conversation` so we can tell whether a "no history" complaint is
about the data, the route, or the Flutter parser.

Run from `backend/` after the app has booted (so the engine is built):

    python -m scripts.diag_conversations

It bootstraps the engine fresh against the current `config.yaml`, so
the data must be reachable from this process too — same host, port,
database, schema, user, password as the app uses.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure `backend/` is on sys.path so `import app...` resolves when run
# as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config_loader import cfg  # noqa: E402
from storage.db import (  # noqa: E402
    create_engine,
    dispose_engine,
    get_session_factory,
)
from storage.models import Message, Session as SessionRow  # noqa: E402


async def main() -> int:
    pg = cfg.database.postgres
    print(
        f"Connecting to {pg.user}@{pg.host}:{pg.port}/{pg.db} "
        f"(schema_name={pg.schema_name!r})..."
    )
    create_engine()
    factory = get_session_factory()
    if factory is None:
        print("ERROR: SessionFactory still None after create_engine().")
        return 1

    try:
        async with factory() as session:
            # 1. How many conversations + messages does Postgres think exist?
            sessions = (
                await session.execute(
                    select(SessionRow).order_by(SessionRow.updated_at.desc())
                )
            ).scalars().all()
            messages = (await session.execute(select(Message))).scalars().all()

            print()
            print(f"sessions: {len(sessions)} row(s)")
            for s in sessions[:20]:
                print(
                    f"  - id={s.id}  title={s.title!r:40}  "
                    f"updated_at={s.updated_at}"
                )
            print()
            print(f"messages: {len(messages)} row(s)")
            for m in messages[:20]:
                preview = (m.content or "")[:60].replace("\n", " ")
                print(
                    f"  - id={m.id}  session_id={m.session_id}  "
                    f"role={m.role}  text={preview!r}"
                )

            # 2. What would GET /api/conversations actually return?
            print()
            print("GET /api/conversations would return:")
            for s in sessions:
                print(
                    {
                        "id": str(s.id),
                        "title": s.title,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    }
                )
    finally:
        await dispose_engine()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
