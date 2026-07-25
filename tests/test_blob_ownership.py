"""Per-user blob/file access (§10.1c).

A signed-in user can fetch a blob only when they own a resource referencing it;
another user can't (even with the exact path). DB-gated.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def _make_engine():
    from storage.db import _build_url, _search_path
    return create_async_engine(
        _build_url(),
        connect_args={"server_settings": {"search_path": _search_path()}})


def _db_ready() -> bool:
    async def _c() -> None:
        eng = _make_engine()
        try:
            async with eng.begin() as conn:
                await conn.execute(text("SELECT 1 FROM messages LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres not reachable")
def test_blob_owner_only(monkeypatch):
    from app.api import routes_blob
    from storage.models import Message
    from storage.models import Session as SessionRow
    from storage.models import User

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    # _owns_blob resolves the factory via storage.db.get_session_factory.
    monkeypatch.setattr("storage.db.get_session_factory", lambda: sf)

    uA, uB = uuid.uuid4(), uuid.uuid4()
    sidA = uuid.uuid4()
    path = f"chat_images/{uuid.uuid4().hex}_shot.png"

    async def _owns_as(uid):
        return await routes_blob._owns_blob(path, uid)

    async def run():
        async with sf() as s:
            s.add_all([User(id=uA, preferences={}), User(id=uB, preferences={})])
            await s.commit()
        async with sf() as s:
            s.add(SessionRow(id=sidA, user_id=uA, title="A", type="chat"))
            await s.commit()
        async with sf() as s:
            # A's message references the blob path in its sources.
            s.add(Message(session_id=sidA, role="user", content="see image",
                          sources={"images": [{"name": "shot.png", "path": path}]}))
            await s.commit()

        assert await _owns_as(uA) is True    # owner can fetch
        assert await _owns_as(uB) is False    # another user CANNOT (the point)

    async def cleanup():
        async with sf() as s:
            obj = await s.get(SessionRow, sidA)
            if obj is not None:
                await s.delete(obj)           # cascades the message
            for uid in (uA, uB):
                u = await s.get(User, uid)
                if u is not None:
                    await s.delete(u)
            await s.commit()
        await eng.dispose()

    async def main():
        try:
            await run()
        finally:
            await cleanup()

    asyncio.run(main())
