"""Email verification flow (§10.1c) — token, template, and register→verify→login.

Hermetic parts always run; the register/verify/login round-trip is DB-gated and
mocks the SMTP send (no real email).
"""
from __future__ import annotations

import asyncio
import time
import types
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import auth_native as an
from app.api import email_sender as es

_SECRET = "email-test-secret-32-bytes-minimum-abcdefgh"


# ── verification token ──────────────────────────────────────────────────────
def test_verify_token_roundtrip(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    uid = str(uuid.uuid4())
    tok = an.mint_verification_token(uid)
    claims = an.verify_verification_token(tok)
    assert claims["sub"] == uid
    assert claims["purpose"] == "email_verify"


def test_session_token_is_not_accepted_as_verification(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    # A real login token (aud=authenticated, no purpose) must be rejected here.
    session_tok = an.mint_token(str(uuid.uuid4()), email="a@b.com")
    with pytest.raises(an.TokenInvalid):
        an.verify_verification_token(session_tok)


def test_verification_token_and_session_verifier_are_disjoint(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    # ...and a verification token must NOT pass as a session token.
    vtok = an.mint_verification_token(str(uuid.uuid4()))
    with pytest.raises(an.AuthError):
        an.verify_native_token(vtok)


def test_expired_verification_token(monkeypatch):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    tok = an.mint_verification_token(str(uuid.uuid4()), ttl_s=1,
                                     now=int(time.time()) - 4000)
    with pytest.raises(an.TokenExpired):
        an.verify_verification_token(tok)


# ── email content + config ──────────────────────────────────────────────────
def test_email_template_has_button_and_link():
    url = "https://host/api/auth/verify?token=xyz"
    html = es.verification_html(url, "Sam")
    assert "Verify email" in html and url in html and "Sam" in html
    assert es.verification_text(url).count(url) >= 1


def test_email_enabled_gating(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_FROM", "SMTP_USER"):
        monkeypatch.delenv(k, raising=False)
    assert es.email_enabled() is False
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_FROM", "me@gmail.com")
    assert es.email_enabled() is True


def test_require_verification_default_and_override(monkeypatch):
    monkeypatch.delenv("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION", raising=False)
    assert an.require_email_verification() is True
    monkeypatch.setenv("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION", "0")
    assert an.require_email_verification() is False


# ── register → verify → login (DB-gated) ────────────────────────────────────
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
                await conn.execute(text("SELECT email_verified FROM users LIMIT 1"))
        finally:
            await eng.dispose()
    try:
        asyncio.run(_c())
        return True
    except Exception:  # noqa: BLE001
        return False


_DB = _db_ready()


@pytest.mark.skipif(not _DB, reason="Postgres not reachable")
def test_register_verify_login_flow(monkeypatch):
    from fastapi import HTTPException
    from app.api import routes_auth as ra
    from app.api.routes_auth import (
        LoginRequest, RegisterRequest, login, register, verify_email,
        verify_status)

    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION", "1")
    # Pretend SMTP is configured so registration proceeds; capture the link
    # instead of sending it.
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_FROM", "no-reply@example.com")
    monkeypatch.setenv("ZAPTHETRICK_PUBLIC_URL", "http://testhost")

    captured: dict = {}

    async def _fake_send(to, url, name=None):
        captured["to"] = to
        captured["url"] = url

    monkeypatch.setattr(es, "send_verification_email", _fake_send)

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    email = f"verify-{uuid.uuid4().hex[:8]}@example.com"
    req = types.SimpleNamespace(base_url="http://testhost/", headers={}, client=types.SimpleNamespace(host="127.0.0.1"))

    async def run():
        async with sf() as s:
            resp = await register(
                RegisterRequest(email=email, password="hunter2!", name="V"),
                req, s)
        assert resp.status == "verification_sent"
        assert resp.token is None
        assert captured["to"] == email
        token = parse_qs(urlparse(captured["url"]).query)["token"][0]

        # login before verifying → 403
        with pytest.raises(HTTPException) as ei:
            async with sf() as s:
                await login(LoginRequest(email=email, password="hunter2!"), req, s)
        assert ei.value.status_code == 403

        # verify-status shows unverified
        async with sf() as s:
            st = await verify_status(email, s)
        assert st.exists and not st.verified

        # hit the verify link → page OK + account confirmed
        async with sf() as s:
            page = await verify_email(token, s)
        assert page.status_code == 200
        async with sf() as s:
            st2 = await verify_status(email, s)
        assert st2.verified

        # login now succeeds
        async with sf() as s:
            tok = await login(LoginRequest(email=email, password="hunter2!"), req, s)
        assert tok.token

    async def cleanup():
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


@pytest.mark.skipif(not _DB, reason="Postgres not reachable")
def test_register_refuses_without_smtp(monkeypatch):
    from fastapi import HTTPException
    from app.api.routes_auth import RegisterRequest, register

    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    monkeypatch.setenv("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION", "1")
    for k in ("SMTP_HOST", "SMTP_FROM", "SMTP_USER"):
        monkeypatch.delenv(k, raising=False)

    eng = _make_engine()
    sf = async_sessionmaker(eng, expire_on_commit=False)
    req = types.SimpleNamespace(base_url="http://testhost/", headers={}, client=types.SimpleNamespace(host="127.0.0.1"))

    async def main():
        try:
            with pytest.raises(HTTPException) as ei:
                async with sf() as s:
                    await register(
                        RegisterRequest(email="x@y.com", password="hunter2!"),
                        req, s)
            assert ei.value.status_code == 503
        finally:
            await eng.dispose()

    asyncio.run(main())
