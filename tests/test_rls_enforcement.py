"""DB-enforced tenant isolation (vNext §10.2 / Phase B3).

Proves the full RLS mechanism end-to-end against a live Postgres: the
`after_begin` hook stamps `SET LOCAL app.user_id` per transaction, and the
policy then makes user A's rows invisible to user B (and to an unscoped
request). DB-gated + robust cleanup (always disables RLS + deletes fixtures) so
a crash can't leave the shared `sessions` table locked down.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import _current_user_id
from storage import tenancy


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


def test_after_begin_hook_sets_the_guc(monkeypatch):
    """The load-bearing, always-run check: the async `after_begin` hook actually
    stamps `SET LOCAL app.user_id` through asyncpg, per transaction. (RLS
    filtering itself can't be observed under a superuser/BYPASSRLS role — see the
    isolation test — but a correct GUC + the unit-tested policy is the guarantee.)
    """
    if not _DB:
        pytest.skip("Postgres not reachable")
    monkeypatch.setenv("TENANT_RLS_ENABLE", "1")
    tenancy.install_rls_session_hook()
    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    u = uuid.uuid4()

    async def guc_as(uid):
        tok = _current_user_id.set(str(uid) if uid else None)
        try:
            async with sf() as s:
                return (await s.execute(
                    text("SELECT current_setting('app.user_id', true)"))).scalar()
        finally:
            _current_user_id.reset(tok)

    async def main():
        try:
            assert await guc_as(u) == str(u)     # signed-in → scoped to that user
            assert (await guc_as(None) or "") == ""  # unscoped → empty (RLS denies)
        finally:
            await eng.dispose()

    asyncio.run(main())


@pytest.mark.skipif(not _DB, reason="Postgres not reachable")
def test_rls_isolates_sessions_between_users(monkeypatch):
    from storage.models import Session as SessionRow
    from storage.models import User

    monkeypatch.setenv("TENANT_RLS_ENABLE", "1")
    tenancy.install_rls_session_hook()  # global + idempotent

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    uA, uB = uuid.uuid4(), uuid.uuid4()
    sidA, sidB = uuid.uuid4(), uuid.uuid4()

    async def _role_bypasses_rls(s) -> bool:
        su = (await s.execute(text("SELECT current_setting('is_superuser')"))).scalar()
        by = (await s.execute(
            text("SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )).scalar()
        return str(su) == "on" or bool(by)

    async def visible_as(uid):
        tok = _current_user_id.set(str(uid) if uid else None)
        try:
            async with sf() as s:
                rows = (await s.execute(
                    select(SessionRow.id).where(
                        SessionRow.id.in_([sidA, sidB]))
                )).scalars().all()
                return set(rows)
        finally:
            _current_user_id.reset(tok)

    async def setup():
        # Insert BEFORE enabling RLS (no WITH CHECK to satisfy yet).
        async with sf() as s:
            s.add_all([User(id=uA, preferences={}), User(id=uB, preferences={})])
            await s.commit()
        async with sf() as s:
            s.add_all([
                SessionRow(id=sidA, user_id=uA, title="A", type="chat"),
                SessionRow(id=sidB, user_id=uB, title="B", type="chat"),
            ])
            await s.commit()

    async def enable_rls():
        async with eng.begin() as conn:  # a Connection → the Session hook stays out
            for stmt in tenancy.rls_policy_statements("sessions"):
                await conn.exec_driver_sql(stmt)

    async def disable_rls():
        async with eng.begin() as conn:
            await conn.exec_driver_sql(
                'DROP POLICY IF EXISTS sessions_tenant_isolation ON "sessions";')
            await conn.exec_driver_sql(
                'ALTER TABLE "sessions" NO FORCE ROW LEVEL SECURITY;')
            await conn.exec_driver_sql(
                'ALTER TABLE "sessions" DISABLE ROW LEVEL SECURITY;')

    async def cleanup_rows():
        async with sf() as s:
            for sid in (sidA, sidB):
                obj = await s.get(SessionRow, sid)
                if obj is not None:
                    await s.delete(obj)
            for uid in (uA, uB):
                u = await s.get(User, uid)
                if u is not None:
                    await s.delete(u)
            await s.commit()

    async def main():
        async with sf() as s:
            bypass = await _role_bypasses_rls(s)
        if bypass:
            # A superuser / BYPASSRLS role ignores policies — filtering can't be
            # asserted here. The GUC mechanism is proven by the test above; the
            # policy SQL by tests/test_tenancy.py. In prod the app connects as a
            # NON-superuser role, where these become hard guarantees.
            pytest.skip(
                "connection role bypasses RLS (superuser) — see the GUC test")
        await setup()
        try:
            await enable_rls()
            # The heart of it: each user sees ONLY their own row; an unscoped
            # request (no user → app.user_id='') sees neither.
            assert await visible_as(uA) == {sidA}
            assert await visible_as(uB) == {sidB}
            assert await visible_as(None) == set()
        finally:
            await disable_rls()      # always, even on assertion failure
            await cleanup_rows()
            await eng.dispose()

    asyncio.run(main())  # a pytest.skip inside propagates out and skips
