"""Native (self-hosted) auth — §10.1c.

The security-critical core (password hashing, HS256 mint/verify, mode logic, the
middleware decision) is tested HERMETICALLY with no DB. register/login/me are
exercised end-to-end but skip-gated on a reachable Postgres (the User row needs
the 0021 columns).
"""
from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import auth as authmod
from app.api import auth_native as an


# ── password hashing ───────────────────────────────────────────────────────
def test_password_hash_roundtrip():
    h = an.hash_password("correct horse battery")
    assert h.startswith("pbkdf2_sha256$")
    assert an.verify_password("correct horse battery", h) is True
    assert an.verify_password("wrong", h) is False


def test_password_hash_is_salted_and_unique():
    a = an.hash_password("same-password")
    b = an.hash_password("same-password")
    assert a != b  # random salt
    assert an.verify_password("same-password", a)
    assert an.verify_password("same-password", b)


def test_verify_password_rejects_garbage():
    assert an.verify_password("x", None) is False
    assert an.verify_password("x", "") is False
    assert an.verify_password("x", "not-a-valid-hash") is False
    assert an.verify_password("", an.hash_password("y")) is False


# ── token mint / verify ─────────────────────────────────────────────────────
# ≥32 bytes so PyJWT doesn't emit InsecureKeyLengthWarning (suite is warn-clean).
_SECRET = "unit-test-secret-please-ignore-32bytes-min-abcdef"


def test_mint_and_verify_roundtrip():
    uid = str(uuid.uuid4())
    tok = _mint(uid, email="a@b.com")
    claims = an.verify_native_token(tok, secret=_SECRET)
    assert claims["sub"] == uid
    assert claims["email"] == "a@b.com"
    assert claims["aud"] == "authenticated"
    assert an.is_native_token(tok) is True


def _mint(uid, **kw):
    # mint_token reads native_secret() for signing; inject via env-free path by
    # monkeypatch-free direct call using the module secret helper isn't possible,
    # so tests set the secret through native_secret by monkeypatch in fixtures.
    import jwt
    now = int(time.time())
    claims = {"sub": uid, "aud": "authenticated", "iss": "zapthetrick",
              "iat": now, "exp": now + 3600, "email": kw.get("email", "a@b.com"),
              "email_verified": True}
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def test_real_mint_token_uses_env_secret(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    uid = str(uuid.uuid4())
    tok = an.mint_token(uid, email="x@y.com")
    claims = an.verify_native_token(tok, secret=_SECRET)
    assert claims["sub"] == uid and claims["email"] == "x@y.com"


def test_expired_token_rejected():
    import jwt
    now = int(time.time())
    tok = jwt.encode(
        {"sub": "u", "aud": "authenticated", "iat": now - 4000,
         "exp": now - 3600}, _SECRET, algorithm="HS256")
    with pytest.raises(an.TokenExpired):
        an.verify_native_token(tok, secret=_SECRET)


def test_wrong_secret_rejected():
    tok = _mint(str(uuid.uuid4()))
    with pytest.raises(an.TokenInvalid):
        an.verify_native_token(
            tok, secret="a-different-secret-also-32-bytes-long-xyzzy")


def test_tampered_token_rejected():
    tok = _mint(str(uuid.uuid4()))
    tampered = tok[:-3] + ("abc" if tok[-3:] != "abc" else "xyz")
    with pytest.raises(an.AuthError):
        an.verify_native_token(tampered, secret=_SECRET)


def test_malformed_token_rejected():
    with pytest.raises(an.TokenInvalid):
        an.verify_native_token("not.a.jwt.at.all", secret=_SECRET)
    with pytest.raises(an.TokenInvalid):
        an.verify_native_token("", secret=_SECRET)


# ── mode / enablement (env-driven) ──────────────────────────────────────────
def test_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("ZAPTHETRICK_AUTH_SECRET", raising=False)
    monkeypatch.delenv("ZAPTHETRICK_AUTH_MODE", raising=False)
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    # No cfg.auth in the default test config → everything off.
    assert an.native_secret() == ""
    assert an.native_enabled() is False
    assert an.auth_enforced() is False


def test_native_enabled_when_secret_set(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    assert an.native_secret() == _SECRET
    assert an.auth_mode() == "native"
    assert an.native_enabled() is True


def test_enforce_flag(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    assert an.auth_enforced() is False        # optional accounts (local)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    assert an.auth_enforced() is True          # login wall (RunPod)


def test_mode_inferred_from_secret(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.delenv("ZAPTHETRICK_AUTH_MODE", raising=False)
    assert an.auth_mode() == "native"          # inferred from a present secret


# ── the middleware decision (hermetic) ──────────────────────────────────────
def test_authorize_optional_allows_anonymous(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    uid, err = authmod.authorize("/api/chat/stream", None)
    assert uid is None and err is None         # anonymous allowed when optional


def test_authorize_enforced_rejects_no_token(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    uid, err = authmod.authorize("/api/chat/stream", None)
    assert uid is None and err and err["status"] == 401
    assert err["code"] == "token_missing"


def test_authorize_accepts_native_token(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    uid_str = str(uuid.uuid4())
    tok = _mint(uid_str)
    uid, err = authmod.authorize("/api/chat/stream", f"Bearer {tok}")
    assert err is None and uid == uid_str


def test_authorize_exempt_path_always_open(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    for p in ("/healthz", "/readyz", "/api/auth/login", "/api/auth/status"):
        uid, err = authmod.authorize(p, None)
        assert uid is None and err is None, p


def test_authorize_present_bad_token_is_rejected_even_when_optional(monkeypatch):
    # A PRESENT but bad token is a client error → 401 even in optional mode, so a
    # signed-in client never silently gets anonymous (device-user) data.
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    uid, err = authmod.authorize("/api/chat/stream", "Bearer garbage.token.x")
    assert uid is None and err and err["status"] == 401


def test_authorize_missing_token_still_anonymous_when_optional(monkeypatch):
    # ...but a MISSING token in optional mode is still legit anonymous.
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    uid, err = authmod.authorize("/api/chat/stream", None)
    assert uid is None and err is None


# ── status endpoint (TestClient, no DB) ─────────────────────────────────────
def _mini_app():
    from fastapi import FastAPI
    from app.api.routes_auth import router
    app = FastAPI()
    app.include_router(router)
    return app


def test_status_off_by_default(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("ZAPTHETRICK_AUTH_SECRET", raising=False)
    monkeypatch.delenv("ZAPTHETRICK_AUTH_MODE", raising=False)
    c = TestClient(_mini_app())
    r = c.get("/api/auth/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "off"
    assert body["enabled"] is False
    assert body["enforced"] is False


def test_status_native_optional(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)
    c = TestClient(_mini_app())
    body = c.get("/api/auth/status").json()
    assert body["mode"] == "native"
    assert body["enabled"] is True
    assert body["enforced"] is False
    assert body["requires_verification"] is False


# ── register / login / me (end-to-end, DB-gated) ────────────────────────────
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
                # Needs the 0021 columns to exist.
                await conn.execute(text("SELECT email, password_hash FROM users LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres w/ 0021 columns not reachable")
def test_register_login_me_end_to_end(monkeypatch):
    """Call the route coroutines directly under one event loop (the sync
    TestClient runs its own loop, which fights asyncpg's loop-bound connections
    — the same reason the other DB integration tests use asyncio.run)."""
    import types

    from fastapi import HTTPException
    from app.api import auth as authmod
    from app.api.routes_auth import (
        LoginRequest, RegisterRequest, login, me, register)

    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    # This test covers the password path, not the email flow — turn verification
    # off so register returns an active account + token directly.
    monkeypatch.setenv("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION", "0")

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    email = f"e2e-{uuid.uuid4().hex[:10]}@example.com"
    req = types.SimpleNamespace(
        base_url="http://testhost/", headers={},
        client=types.SimpleNamespace(host="127.0.0.1"))

    async def run():
        async with sf() as s:
            resp = await register(
                RegisterRequest(email=email, password="hunter2!", name="E2E"),
                req, s)
        assert resp.status == "active"
        assert resp.user.email == email and resp.token
        tok = resp.token

        # duplicate → 409
        with pytest.raises(HTTPException) as ei:
            async with sf() as s:
                await register(
                    RegisterRequest(email=email, password="hunter2!"), req, s)
        assert ei.value.status_code == 409

        # wrong password → 401
        with pytest.raises(HTTPException) as ei2:
            async with sf() as s:
                await login(LoginRequest(email=email, password="nope"), req, s)
        assert ei2.value.status_code == 401

        # right password → token
        async with sf() as s:
            assert (await login(
                LoginRequest(email=email, password="hunter2!"), req, s)).token

        # /me with the verified user context set
        claims = authmod.verify_any_token(tok)
        ctx = authmod._current_user_id.set(claims["sub"])
        try:
            async with sf() as s:
                meout = await me(s)
            assert meout.email == email
        finally:
            authmod._current_user_id.reset(ctx)

    async def cleanup():
        from sqlalchemy import func, select
        from storage.models import User
        async with sf() as s:
            u = (await s.execute(
                select(User).where(func.lower(User.email) == email)
            )).scalar_one_or_none()
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
