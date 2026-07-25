"""Google OAuth core (§10.1c) — config gating, consent URL, code exchange.

Hermetic: the Google token endpoint is faked and id_token verification is
injected, so no network / real Google client is needed.
"""
from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from app.api import auth_oauth


@pytest.fixture(autouse=True)
def _clear_hook():
    yield
    auth_oauth._verify_hook = None


def test_google_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    assert auth_oauth.google_enabled() is False


def test_google_enabled_with_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    assert auth_oauth.google_enabled() is True
    assert auth_oauth.google_client_id().startswith("cid")


def test_build_auth_url_has_required_params(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    url = auth_oauth.build_auth_url("http://127.0.0.1:8123", "state-xyz")
    p = urlparse(url)
    q = parse_qs(p.query)
    assert p.netloc == "accounts.google.com"
    assert q["client_id"] == ["cid"]
    assert q["redirect_uri"] == ["http://127.0.0.1:8123"]
    assert q["response_type"] == ["code"]
    assert q["state"] == ["state-xyz"]
    assert "openid" in q["scope"][0]


class _FakeResp:
    status_code = 200

    def json(self):
        return {"id_token": "fake.jwt.token", "access_token": "at"}


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, data=None):
        # Assert we send the code + client creds to Google's token endpoint.
        assert url == auth_oauth.GOOGLE_TOKEN_URL
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "the-code"
        return _FakeResp()


def test_exchange_code_returns_identity(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setattr(auth_oauth.httpx, "AsyncClient", _FakeClient)
    auth_oauth._verify_hook = lambda t: {
        "sub": "google-uid-1",
        "email": "Person@Example.com",
        "name": "A Person",
        "email_verified": True,
        "iss": "https://accounts.google.com",
    }
    ident = asyncio.run(
        auth_oauth.exchange_code("the-code", "http://127.0.0.1:8123"))
    assert ident.sub == "google-uid-1"
    assert ident.email == "person@example.com"  # lowercased
    assert ident.name == "A Person"
    assert ident.email_verified is True


class _BadResp:
    status_code = 400

    def json(self):
        return {"error": "invalid_grant"}


class _BadClient(_FakeClient):
    async def post(self, url, data=None):
        return _BadResp()


def test_exchange_code_rejects_bad_grant(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setattr(auth_oauth.httpx, "AsyncClient", _BadClient)
    with pytest.raises(auth_oauth.TokenInvalid):
        asyncio.run(auth_oauth.exchange_code("bad", "http://127.0.0.1:8123"))
