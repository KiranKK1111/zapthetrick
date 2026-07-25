"""Per-user model catalog + fallback (§10.1c).

Each user's `seed_provider` builds THEIR own catalog rows; the router-style scope
returns only that user's models (+ the shared local floor), never another's.
DB-gated on the 0023 `user_id` column.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import _current_user_id
from app.llm.catalog import seed_provider, seed_rows


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
                await conn.execute(text("SELECT user_id FROM llm_models LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres w/ 0023 column not reachable")
def test_model_catalog_isolated_per_user(monkeypatch):
    from storage.models import LLMModel, User

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: sf)

    plat = next(r["platform"] for r in seed_rows())  # a platform with curated rows
    uA, uB = uuid.uuid4(), uuid.uuid4()

    async def seed_as(uid):
        tok = _current_user_id.set(str(uid))
        try:
            return await seed_provider(plat)
        finally:
            _current_user_id.reset(tok)

    async def models_for(uid):
        async with sf() as s:
            return (await s.execute(
                select(LLMModel.id).where(LLMModel.user_id == uid,
                                          LLMModel.platform == plat)
            )).scalars().all()

    async def router_scope_ids(uid):
        # Mirror the router's filter: this user's rows + the shared local floor.
        async with sf() as s:
            return set((await s.execute(
                select(LLMModel.id).where(
                    or_(LLMModel.user_id == uid, LLMModel.platform == "local"))
            )).scalars().all())

    async def run():
        async with sf() as s:
            s.add_all([User(id=uA, preferences={}), User(id=uB, preferences={})])
            await s.commit()

        addedA = await seed_as(uA)
        addedB = await seed_as(uB)
        assert addedA > 0 and addedB > 0        # each user seeded their own rows

        idsA, idsB = set(await models_for(uA)), set(await models_for(uB))
        assert idsA and idsB
        assert idsA.isdisjoint(idsB)            # separate catalog rows per user

        scopeA = await router_scope_ids(uA)
        assert idsA <= scopeA                   # A's router sees A's models
        assert scopeA.isdisjoint(idsB)          # ...and NEVER B's

    async def cleanup():
        from storage.models import User as U
        async with sf() as s:
            await s.execute(
                LLMModel.__table__.delete().where(LLMModel.user_id.in_([uA, uB])))
            for uid in (uA, uB):
                u = await s.get(U, uid)
                if u is not None:
                    await s.delete(u)   # cascades any leftover model/fallback rows
            await s.commit()
        await eng.dispose()

    async def main():
        try:
            await run()
        finally:
            await cleanup()

    asyncio.run(main())
