"""Per-user chat / live sessions (§10.1c) — owner-stamp + scoped list/delete.

DB-gated. Proves a new session is auto-stamped with the request user, that a
scoped list returns only that user's sessions, and that one user can't delete
another's.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import _current_user_id
from storage import tenancy
from storage.repos.session_repo import SessionRepo


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
                await conn.execute(text("SELECT user_id FROM sessions LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres not reachable")
def test_sessions_owner_stamped_and_isolated(monkeypatch):
    from storage.models import User

    tenancy.install_owner_stamp_hook()  # global + idempotent
    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    uA, uB = uuid.uuid4(), uuid.uuid4()

    async def as_user(uid, fn):
        tok = _current_user_id.set(str(uid) if uid else None)
        try:
            async with sf() as s:
                out = await fn(SessionRepo(s), s)
                await s.commit()
                return out
        finally:
            _current_user_id.reset(tok)

    async def run():
        async with sf() as s:
            s.add_all([User(id=uA, preferences={}), User(id=uB, preferences={})])
            await s.commit()

        # Create WITHOUT user_id — the before_insert hook stamps the request user.
        rowA = await as_user(uA, lambda r, s: r.create(type="chat", title="A1"))
        rowB = await as_user(uB, lambda r, s: r.create(type="chat", title="B1"))
        assert rowA.user_id == uA          # auto-owned
        assert rowB.user_id == uB
        sidA, sidB = rowA.id, rowB.id

        # Scoped list: each user sees only their own.
        listA = await as_user(uA, lambda r, s: r.list(user_id=uA))
        listB = await as_user(uB, lambda r, s: r.list(user_id=uB))
        assert sidA in {x.id for x in listA} and sidB not in {x.id for x in listA}
        assert sidB in {x.id for x in listB} and sidA not in {x.id for x in listB}

        # A cannot delete B's session; B can.
        okCross = await as_user(uA, lambda r, s: r.delete(sidB, user_id=uA))
        assert okCross is False
        okOwn = await as_user(uB, lambda r, s: r.delete(sidB, user_id=uB))
        assert okOwn is True

    async def cleanup():
        from storage.models import Session as SessionRow
        async with sf() as s:
            await s.execute(
                SessionRow.__table__.delete().where(
                    SessionRow.user_id.in_([uA, uB])))
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
