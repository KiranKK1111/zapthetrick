"""Auth middleware + RequestContext (vNext §10.1b)."""
from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from app.api import auth as M
from app.api import auth_jwt as A

_KID = "mw-kid"


def _keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk.update(kid=_KID, alg="RS256", use="sig")
    return priv, A.JwksCache(fetch=lambda: {"keys": [jwk]})


def _sign(priv, **over):
    c = {"sub": "abc-123", "aud": "authenticated",
         "exp": int(time.time()) + 3600, "email_confirmed_at": "2024-01-01"}
    c.update(over)
    return jwt.encode(c, priv, algorithm="RS256", headers={"kid": _KID})


def _app(cache):
    app = FastAPI()
    app.add_middleware(M.AuthMiddleware, jwks=cache)

    @app.get("/healthz")
    def _h():
        return {"ok": True}

    @app.get("/protected")
    def _p():
        return {"uid": M.current_user_id()}

    return app


# ── pure helpers ─────────────────────────────────────────────────────────
def test_bearer_token_parsing():
    assert M.bearer_token("Bearer xyz") == "xyz"
    assert M.bearer_token("bearer xyz") == "xyz"
    assert M.bearer_token("xyz") is None
    assert M.bearer_token("Bearer ") is None
    assert M.bearer_token(None) is None


def test_exempt_and_auth_off_allow_without_token(monkeypatch):
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    assert M.authorize("/healthz", None) == (None, None)      # exempt
    monkeypatch.setattr(A, "auth_enabled", lambda: False)
    assert M.authorize("/protected", None) == (None, None)    # auth off


# ── middleware end-to-end ────────────────────────────────────────────────
def test_healthz_open_even_with_auth_on(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    c = TestClient(_app(cache))
    assert c.get("/healthz").status_code == 200


def test_protected_requires_a_token(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    r = TestClient(_app(cache)).get("/protected")
    assert r.status_code == 401
    assert r.json()["code"] == "token_missing"


def test_protected_accepts_a_valid_token_and_sets_context(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    r = TestClient(_app(cache)).get(
        "/protected", headers={"Authorization": f"Bearer {_sign(priv)}"})
    assert r.status_code == 200
    assert r.json()["uid"] == "abc-123"


def test_expired_token_is_401_typed(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    tok = _sign(priv, exp=int(time.time()) - 300)
    r = TestClient(_app(cache)).get(
        "/protected", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert r.json()["code"] == "token_expired"


def test_unverified_email_is_403_typed(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    monkeypatch.setattr(A, "require_verified_email", lambda: True)
    tok = _sign(priv, email_confirmed_at=None, email_verified=False)
    r = TestClient(_app(cache)).get(
        "/protected", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    assert r.json()["code"] == "email_unverified"


def test_auth_off_is_passthrough(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: False)
    r = TestClient(_app(cache)).get("/protected")   # no token, but auth off
    assert r.status_code == 200
    assert r.json()["uid"] is None


# ── WS handshake ─────────────────────────────────────────────────────────
def test_ws_auth_off_is_noop(monkeypatch):
    monkeypatch.setattr(A, "auth_enabled", lambda: False)
    assert M.authenticate_ws_first_frame({"anything": 1}) == (None, None)


def test_ws_first_frame_must_be_auth(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    uid, err = M.authenticate_ws_first_frame({"type": "start"}, jwks=cache)
    assert uid is None and err["code"] == "auth_required"


def test_ws_valid_auth_frame(monkeypatch):
    priv, cache = _keypair()
    monkeypatch.setattr(A, "auth_enabled", lambda: True)
    uid, err = M.authenticate_ws_first_frame(
        {"type": "auth", "token": _sign(priv)}, jwks=cache)
    assert err is None and uid == "abc-123"
