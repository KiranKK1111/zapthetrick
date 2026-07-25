"""Auth hardening (§10.1c) — rate limiting, password-reset tokens, WS auth."""
from __future__ import annotations

import time
import uuid

import jwt
import pytest

from app.api import auth as authmod
from app.api import auth_native as an
from app.api import rate_limit

_SECRET = "hardening-test-secret-32-bytes-minimum-abcdefgh"


def _native_env(monkeypatch, *, enforce=False):
    monkeypatch.setenv("ZAPTHETRICK_AUTH_SECRET", _SECRET)
    monkeypatch.setenv("ZAPTHETRICK_AUTH_MODE", "native")
    if enforce:
        monkeypatch.setenv("ZAPTHETRICK_AUTH_ENFORCE", "1")
    else:
        monkeypatch.delenv("ZAPTHETRICK_AUTH_ENFORCE", raising=False)


def _session_token(uid: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": uid, "aud": "authenticated", "iat": now, "exp": now + 3600,
         "email_verified": True}, _SECRET, algorithm="HS256")


# ── rate limiter ─────────────────────────────────────────────────────────────
def test_rate_limit_blocks_after_max():
    rate_limit.reset_for_tests()
    t = 1000.0
    for _ in range(3):
        assert rate_limit.check_rate("k", max_attempts=3, window_s=60, now=t)
    assert rate_limit.check_rate("k", max_attempts=3, window_s=60, now=t) is False


def test_rate_limit_window_slides():
    rate_limit.reset_for_tests()
    for _ in range(3):
        rate_limit.check_rate("k2", max_attempts=3, window_s=60, now=1000.0)
    assert rate_limit.check_rate("k2", max_attempts=3, window_s=60, now=1000) is False
    # After the window passes, attempts are allowed again.
    assert rate_limit.check_rate("k2", max_attempts=3, window_s=60, now=1100.0)


def test_rate_limit_keys_are_independent():
    rate_limit.reset_for_tests()
    assert rate_limit.check_rate("a", max_attempts=1, window_s=60, now=1)
    assert rate_limit.check_rate("a", max_attempts=1, window_s=60, now=1) is False
    assert rate_limit.check_rate("b", max_attempts=1, window_s=60, now=1)  # other key ok


# ── password-reset token ─────────────────────────────────────────────────────
def test_reset_token_roundtrip(monkeypatch):
    _native_env(monkeypatch)
    uid = str(uuid.uuid4())
    tok = an.mint_reset_token(uid, pw_hash="pbkdf2_sha256$abc")
    claims = an.verify_reset_token(tok)
    assert claims["sub"] == uid and claims["purpose"] == "password_reset"
    assert "pwc" in claims


def test_reset_token_rejects_session_token(monkeypatch):
    _native_env(monkeypatch)
    with pytest.raises(an.TokenInvalid):
        an.verify_reset_token(_session_token(str(uuid.uuid4())))


def test_reset_token_and_session_disjoint(monkeypatch):
    _native_env(monkeypatch)
    rtok = an.mint_reset_token(str(uuid.uuid4()))
    with pytest.raises(an.AuthError):
        an.verify_native_token(rtok)  # a reset token can't log you in


def test_reset_token_expired(monkeypatch):
    _native_env(monkeypatch)
    tok = an.mint_reset_token(str(uuid.uuid4()), now=int(time.time()) - 4000,
                              ttl_s=1)
    with pytest.raises(an.TokenExpired):
        an.verify_reset_token(tok)


def test_pw_fingerprint_changes_with_hash():
    assert an.pw_hash_fingerprint("a") != an.pw_hash_fingerprint("b")
    assert an.pw_hash_fingerprint("a") == an.pw_hash_fingerprint("a")


# ── password strength ────────────────────────────────────────────────────────
def test_password_problem_rules():
    assert an.password_problem("short1") is not None       # < 8 chars
    assert "8 characters" in an.password_problem("abc1")
    assert an.password_problem("allletters") is not None    # no digit
    assert an.password_problem("12345678") is not None      # no letter
    assert an.password_problem("hunter2!") is None          # ok (8, letter, digit)
    assert an.password_problem("GoodPass1") is None


# ── WebSocket auth ───────────────────────────────────────────────────────────
def test_ws_optional_allows_anonymous(monkeypatch):
    _native_env(monkeypatch, enforce=False)
    uid, err = authmod.authenticate_ws(None)
    assert uid is None and err is None


def test_ws_optional_scopes_a_valid_token(monkeypatch):
    _native_env(monkeypatch, enforce=False)
    u = str(uuid.uuid4())
    uid, err = authmod.authenticate_ws(_session_token(u))
    assert err is None and uid == u


def test_ws_optional_present_bad_token_is_rejected(monkeypatch):
    # A present-but-bad token is rejected even in optional mode (no silent
    # anonymous scoping); only a MISSING token stays anonymous.
    _native_env(monkeypatch, enforce=False)
    uid, err = authmod.authenticate_ws("garbage.token.x")
    assert uid is None and err is not None


def test_ws_enforced_rejects_no_token(monkeypatch):
    _native_env(monkeypatch, enforce=True)
    uid, err = authmod.authenticate_ws(None)
    assert uid is None and err and err["code"] == "auth_required"


def test_ws_enforced_rejects_bad_token(monkeypatch):
    _native_env(monkeypatch, enforce=True)
    uid, err = authmod.authenticate_ws("garbage.token.x")
    assert uid is None and err is not None


def test_ws_enforced_accepts_valid_token(monkeypatch):
    _native_env(monkeypatch, enforce=True)
    u = str(uuid.uuid4())
    uid, err = authmod.authenticate_ws(_session_token(u))
    assert err is None and uid == u
