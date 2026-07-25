"""Supabase JWT verification core (vNext §10.1) — hermetic, self-signed keys."""
from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.api import auth_jwt as A

_KID = "test-kid-1"


def _keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk.update(kid=_KID, alg="RS256", use="sig")
    return priv, {"keys": [jwk]}


def _sign(priv, claims, *, kid=_KID, alg="RS256"):
    return jwt.encode(claims, priv, algorithm=alg, headers={"kid": kid})


def _valid_claims(**over):
    c = {
        "sub": "11111111-2222-3333-4444-555555555555",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
        "email_confirmed_at": "2024-01-01T00:00:00Z",
    }
    c.update(over)
    return c


def test_valid_token_verifies_and_returns_sub():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    claims = A.verify_token(_sign(priv, _valid_claims()), jwks=cache)
    assert A.user_id_from_claims(claims) == "11111111-2222-3333-4444-555555555555"


def test_expired_token_raises_token_expired():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    # Beyond the 60 s clock-skew leeway so it's genuinely expired.
    tok = _sign(priv, _valid_claims(exp=int(time.time()) - 300))
    with pytest.raises(A.TokenExpired):
        A.verify_token(tok, jwks=cache)


def test_wrong_audience_is_rejected():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    tok = _sign(priv, _valid_claims(aud="some-other-app"))
    with pytest.raises(A.TokenInvalid):
        A.verify_token(tok, jwks=cache)


def test_tampered_signature_is_rejected():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    tok = _sign(priv, _valid_claims())
    tampered = tok[:-4] + ("aaaa" if not tok.endswith("aaaa") else "bbbb")
    with pytest.raises(A.TokenInvalid):
        A.verify_token(tampered, jwks=cache)


def test_unknown_kid_is_rejected():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    tok = _sign(priv, _valid_claims(), kid="not-the-real-kid")
    with pytest.raises(A.TokenInvalid):
        A.verify_token(tok, jwks=cache)


def test_malformed_token_is_rejected():
    priv, jwks = _keypair()
    cache = A.JwksCache(fetch=lambda: jwks)
    with pytest.raises(A.TokenInvalid):
        A.verify_token("not.a.jwt.at.all", jwks=cache)
    with pytest.raises(A.TokenInvalid):
        A.verify_token("", jwks=cache)


def test_email_verification_enforced(monkeypatch):
    monkeypatch.setattr(A, "require_verified_email", lambda: True)
    with pytest.raises(A.EmailUnverified):
        A.enforce_email_verified({"sub": "x"})               # no confirmation
    A.enforce_email_verified({"email_confirmed_at": "2024-..."})  # ok
    A.enforce_email_verified({"email_verified": True})            # ok


def test_email_verification_can_be_disabled(monkeypatch):
    monkeypatch.setattr(A, "require_verified_email", lambda: False)
    A.enforce_email_verified({"sub": "x"})   # no raise


def test_kid_miss_triggers_a_refresh():
    priv, jwks = _keypair()
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return jwks     # the key only becomes known after a fetch

    cache = A.JwksCache(fetch=_fetch)
    # First get with an empty cache fetches; the token then verifies.
    claims = A.verify_token(_sign(priv, _valid_claims()), jwks=cache)
    assert claims["sub"]
    assert calls["n"] >= 1


def test_auth_disabled_by_default():
    assert A.auth_enabled() is False
