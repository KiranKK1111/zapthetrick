"""Per-user provider API keys (§10.1c/§10.2).

The scope helper is tested hermetically; true isolation (user A can't see user
B's keys, anonymous sees only the legacy NULL-owned keys) is DB-gated.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import _current_user_id
from app.llm import keys as keys_repo
from app.llm import router as router_mod


# ── hermetic: the scope helpers read the request ContextVar ─────────────────
def test_scope_user_none_when_unset():
    assert keys_repo._scope_user() is None
    assert router_mod._route_user_id() is None


def test_scope_user_reads_contextvar():
    u = uuid.uuid4()
    tok = _current_user_id.set(str(u))
    try:
        assert keys_repo._scope_user() == u
        assert router_mod._route_user_id() == u
    finally:
        _current_user_id.reset(tok)


def test_scope_user_bad_value_is_none():
    tok = _current_user_id.set("not-a-uuid")
    try:
        assert keys_repo._scope_user() is None
    finally:
        _current_user_id.reset(tok)


# ── DB-gated: real isolation across users ───────────────────────────────────
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
                await conn.execute(text("SELECT user_id FROM llm_api_keys LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres w/ 0022 column not reachable")
def test_keys_isolated_per_user(monkeypatch):
    from storage.models import LLMApiKey, User

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    # keys.py did `from storage.db import get_session_factory`, so patch the name
    # in ITS namespace (patching storage.db wouldn't rebind the import).
    monkeypatch.setattr("app.llm.keys.get_session_factory", lambda: sf)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: sf)

    uA, uB = uuid.uuid4(), uuid.uuid4()

    def _mkkey(user_id, plat):
        # Insert directly (bypass add_key's encryption); list_keys tolerates
        # undecryptable rows by masking, so it still returns them.
        return LLMApiKey(user_id=user_id, platform=plat, label="t",
                         encrypted_key="x", iv="x", auth_tag="x",
                         status="unknown", enabled=True)

    async def run():
        async with sf() as s:
            s.add_all([User(id=uA, preferences={"t": 1}),
                       User(id=uB, preferences={"t": 1})])
            await s.commit()  # parents first (FK on llm_api_keys.user_id)
        async with sf() as s:
            s.add_all([
                _mkkey(uA, "openai"), _mkkey(uA, "groq"),
                _mkkey(uB, "openai"),
                _mkkey(None, "cerebras"),  # legacy global
            ])
            await s.commit()

        async def _list_as(uid):
            tok = _current_user_id.set(str(uid) if uid else None)
            try:
                return await keys_repo.list_keys()
            finally:
                _current_user_id.reset(tok)

        a = await _list_as(uA)
        b = await _list_as(uB)
        anon = await _list_as(None)
        assert {k.platform for k in a} == {"openai", "groq"}   # A sees only A
        assert {k.platform for k in b} == {"openai"}            # B sees only B
        # anon sees the legacy NULL-owned keys (incl. any pre-existing dev keys)
        # but NEVER a user's key — that's the isolation guarantee.
        a_ids, b_ids = {k.id for k in a}, {k.id for k in b}
        anon_ids = {k.id for k in anon}
        assert "cerebras" in {k.platform for k in anon}
        assert not (a_ids & anon_ids) and not (b_ids & anon_ids)

        # A cannot delete B's key (scoped delete is a no-op across users).
        bkey_id = b[0].id
        tokA = _current_user_id.set(str(uA))
        try:
            await keys_repo.delete_key(bkey_id)
        finally:
            _current_user_id.reset(tokA)
        assert len(await _list_as(uB)) == 1  # B's key survived A's delete

    async def cleanup():
        async with sf() as s:
            for uid in (uA, uB):
                await s.execute(
                    LLMApiKey.__table__.delete().where(LLMApiKey.user_id == uid))
                u = await s.get(User, uid)
                if u is not None:
                    await s.delete(u)
            await s.execute(
                LLMApiKey.__table__.delete().where(
                    LLMApiKey.platform == "cerebras",
                    LLMApiKey.user_id.is_(None)))
            await s.commit()
        await eng.dispose()

    async def main():
        try:
            await run()
        finally:
            await cleanup()

    asyncio.run(main())
