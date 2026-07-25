"""Supabase JWT verification core (vNext §10.1).

The pod NEVER mints tokens or stores passwords — it only VERIFIES the JWTs
Supabase issues. Verification is offline and fast: fetch Supabase's JWKS once
(cached by ``kid``, refreshed on a miss), then verify signatures locally with
the public key (~50 µs), so there is no per-request round-trip to Supabase.

This module is pure (no FastAPI) so the crypto is unit-testable with a
self-signed keypair. The FastAPI middleware (auth.py) wraps it.

Flag-gated: ``auth.enabled`` defaults OFF → the app keeps its device-identity
behaviour byte-for-byte. Fail-CLOSED when on: an unverifiable token is rejected,
never allowed through.
"""
from __future__ import annotations

import logging
import threading
import time

import httpx
import jwt
from jwt import PyJWKSet

log = logging.getLogger(__name__)


class AuthError(Exception):
    """Base — a token that must be rejected. ``code`` maps to a typed FE state."""

    code = "auth_error"


class TokenInvalid(AuthError):
    code = "token_invalid"


class TokenExpired(AuthError):
    code = "token_expired"


class EmailUnverified(AuthError):
    code = "email_unverified"


# ── config ───────────────────────────────────────────────────────────────
def auth_enabled() -> bool:
    """Whether JWT auth is enforced. Default OFF → device-identity unchanged."""
    try:
        from app.core.config_loader import cfg
        a = getattr(cfg, "auth", None)
        return bool(getattr(a, "enabled", False)) if a is not None else False
    except Exception:  # noqa: BLE001
        return False


def require_verified_email() -> bool:
    try:
        from app.core.config_loader import cfg
        a = getattr(cfg, "auth", None)
        v = getattr(a, "require_verified_email", True) if a is not None else True
        return True if v is None else bool(v)
    except Exception:  # noqa: BLE001
        return True


def supabase_url() -> str:
    try:
        from app.core.config_loader import cfg
        a = getattr(cfg, "auth", None)
        u = getattr(a, "supabase_url", "") if a is not None else ""
        return str(u or "").rstrip("/")
    except Exception:  # noqa: BLE001
        return ""


def _jwks_urls() -> list[str]:
    base = supabase_url()
    if not base:
        return []
    # Newer Supabase serves standard JWKS; keep the legacy path as a fallback.
    return [f"{base}/auth/v1/.well-known/jwks.json", f"{base}/auth/v1/keys"]


# ── JWKS cache ───────────────────────────────────────────────────────────
class JwksCache:
    """Caches the provider's public keys by ``kid``. Refreshes on a miss, at
    most once per ``min_refresh_s`` so a bad/forged ``kid`` can't hammer the
    provider. Injectable (``fetch``) for tests."""

    def __init__(self, *, min_refresh_s: float = 30.0, fetch=None):
        self._lock = threading.RLock()
        self._keys: dict[str, object] = {}   # kid -> PyJWK signing key
        self._last_refresh = 0.0
        self._min_refresh_s = min_refresh_s
        self._fetch = fetch or self._http_fetch

    @staticmethod
    def _http_fetch() -> dict:
        for url in _jwks_urls():
            try:
                with httpx.Client(timeout=10.0) as c:
                    r = c.get(url)
                if r.status_code == 200 and isinstance(r.json(), dict):
                    return r.json()
            except Exception:  # noqa: BLE001 — try the next url / give up
                continue
        return {"keys": []}

    def _load(self, jwks: dict) -> None:
        try:
            ks = PyJWKSet.from_dict(jwks)
            self._keys = {k.key_id: k for k in ks.keys if k.key_id}
        except Exception as exc:  # noqa: BLE001
            log.warning("jwks: could not parse key set: %s", exc)

    def refresh(self, jwks: dict | None = None) -> None:
        with self._lock:
            self._last_refresh = time.monotonic()
            self._load(jwks if jwks is not None else self._fetch())

    def get(self, kid: str):
        with self._lock:
            key = self._keys.get(kid)
            if key is not None:
                return key
            stale = (time.monotonic() - self._last_refresh) > self._min_refresh_s
            if not self._keys or stale:
                # refresh at most once per window on a miss
                self._last_refresh = time.monotonic()
                self._load(self._fetch())
                return self._keys.get(kid)
            return None


_default_cache: JwksCache | None = None


def default_cache() -> JwksCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = JwksCache()
    return _default_cache


def reset_for_tests() -> None:
    global _default_cache
    _default_cache = None


# ── verification ─────────────────────────────────────────────────────────
def verify_token(token: str, *, jwks: JwksCache | None = None,
                 audience: str = "authenticated", leeway: int = 60) -> dict:
    """Verify a Supabase JWT and return its claims. Raises a typed ``AuthError``
    for every rejection reason."""
    if not token or token.count(".") != 2:
        raise TokenInvalid("malformed token")
    cache = jwks or default_cache()
    try:
        header = jwt.get_unverified_header(token)
    except Exception as exc:  # noqa: BLE001
        raise TokenInvalid(f"bad header: {exc}") from exc
    kid = header.get("kid")
    alg = header.get("alg", "RS256")
    if not kid:
        raise TokenInvalid("no kid in header")
    signing_key = cache.get(kid)
    if signing_key is None:
        raise TokenInvalid("unknown signing key (kid)")
    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[alg],
            audience=audience,
            leeway=leeway,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired("token expired") from exc
    except jwt.InvalidAudienceError as exc:
        raise TokenInvalid("wrong audience") from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"invalid token: {exc}") from exc
    return claims


def enforce_email_verified(claims: dict) -> None:
    """Reject a token whose email isn't confirmed (when required). Supabase puts
    ``email_confirmed_at`` on the user and mirrors it into ``user_metadata`` /
    the top-level ``email_verified`` claim depending on version — accept any."""
    if not require_verified_email():
        return
    if claims.get("email_confirmed_at"):
        return
    if claims.get("email_verified") is True:
        return
    md = claims.get("user_metadata") or {}
    if isinstance(md, dict) and (md.get("email_verified") is True
                                 or md.get("email_confirmed_at")):
        return
    raise EmailUnverified("email not verified")


def user_id_from_claims(claims: dict) -> str | None:
    """The Supabase user id (the ``sub`` UUID) — the tenant key (§10.2)."""
    sub = claims.get("sub")
    return str(sub) if sub else None


__all__ = [
    "AuthError", "TokenInvalid", "TokenExpired", "EmailUnverified",
    "JwksCache", "verify_token", "enforce_email_verified",
    "user_id_from_claims", "auth_enabled", "require_verified_email",
    "supabase_url", "default_cache", "reset_for_tests",
]
