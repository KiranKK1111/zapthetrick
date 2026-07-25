"""Backend-native accounts (vNext §10.1c) — self-hosted email/password auth.

Unlike the Supabase path (auth_jwt.py verifies *someone else's* RS256 JWTs), this
module lets the ZapTheTrick backend BE the identity provider: it hashes passwords
(PBKDF2, stdlib — no bcrypt/argon dep) and mints/verifies its own HS256 JWTs with
a server secret. That means accounts work with **zero external dependency**, the
same way in a local desktop backend and on an exposed RunPod pod (where they're
what actually secures the open port).

The secret is read from `ZAPTHETRICK_AUTH_SECRET` (env, for the baked RunPod
image) or `cfg.auth.jwt_secret`. With no secret, native auth is OFF and the app
keeps its device-identity behaviour byte-for-byte.

Verification reuses the typed errors from auth_jwt so the middleware handles a
native token and a Supabase token through one code path.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

import jwt

from app.api.auth_jwt import AuthError, TokenExpired, TokenInvalid

# ── config (env-first so the RunPod image just sets an env var) ────────────
_ENV_SECRET = "ZAPTHETRICK_AUTH_SECRET"
_ENV_MODE = "ZAPTHETRICK_AUTH_MODE"        # "native" | "supabase" | "off"
_ENV_ENFORCE = "ZAPTHETRICK_AUTH_ENFORCE"  # "1" → require login (gate the pod)

_TOKEN_TTL_S = 60 * 60 * 24 * 30  # 30 days; the client silently re-logins after
_ISSUER = "zapthetrick"
_AUDIENCE = "authenticated"


def _cfg_auth():
    try:
        from app.core.config_loader import cfg
        return getattr(cfg, "auth", None)
    except Exception:  # noqa: BLE001
        return None


def native_secret() -> str:
    """The HMAC secret for signing native JWTs. Env wins over config."""
    env = os.environ.get(_ENV_SECRET, "")
    if env:
        return env
    a = _cfg_auth()
    return str(getattr(a, "jwt_secret", "") or "") if a is not None else ""


def auth_mode() -> str:
    """`native` | `supabase` | `off`. Env wins; else config; else inferred."""
    env = (os.environ.get(_ENV_MODE, "") or "").strip().lower()
    if env in {"native", "supabase", "off"}:
        return env
    a = _cfg_auth()
    m = (str(getattr(a, "mode", "") or "").strip().lower()) if a is not None else ""
    if m in {"native", "supabase", "off"}:
        return m
    # Inference: a native secret present → native; a supabase_url present →
    # supabase; otherwise off.
    if native_secret():
        return "native"
    if a is not None and getattr(a, "supabase_url", ""):
        return "supabase"
    return "off"


def native_enabled() -> bool:
    """True when this backend should host its own accounts."""
    return auth_mode() == "native" and bool(native_secret())


def require_email_verification() -> bool:
    """Whether a new account must confirm its email before it can log in.
    Default ON (user's choice). Set ``ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION=0``
    to disable — e.g. pure local dev with no SMTP configured."""
    v = os.environ.get("ZAPTHETRICK_REQUIRE_EMAIL_VERIFICATION")
    if v is not None:
        return v not in ("0", "false", "no")
    return True


def public_url() -> str:
    """The externally-reachable base URL of THIS backend, for building links in
    outgoing email (verify link). Empty → the route derives it from the request
    host, which is correct for local + most single-hop deploys."""
    return (os.environ.get("ZAPTHETRICK_PUBLIC_URL", "") or "").rstrip("/")


def auth_enforced() -> bool:
    """Whether an unauthenticated request is REJECTED (a real login wall).

    On RunPod set enforce=1 → the exposed pod requires a login. In local mode
    leave it off → accounts are OPTIONAL (you can create one, but the app still
    works as device-identity), so today's desktop UX is preserved.
    """
    if os.environ.get(_ENV_ENFORCE) == "1":
        return True
    a = _cfg_auth()
    if a is not None and bool(getattr(a, "enforce", False)):
        return True
    return False


# ── password hashing (PBKDF2-HMAC-SHA256, stdlib) ──────────────────────────
_PBKDF2_ROUNDS = 210_000  # OWASP 2023 floor for PBKDF2-SHA256


def hash_password(password: str, *, rounds: int = _PBKDF2_ROUNDS) -> str:
    if not password:
        raise ValueError("empty password")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${dk.hex()}"


def password_problem(password: str) -> str | None:
    """A human-readable reason the password is too weak, or None if it's fine.
    Reasonable baseline: >=8 chars with at least one letter and one digit."""
    if len(password or "") < 8:
        return "Password must be at least 8 characters."
    if not any(c.isalpha() for c in password):
        return "Password must contain a letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain a number."
    return None


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or not password:
        return False
    try:
        scheme, rounds_s, salt_hex, hash_hex = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(dk, expected)


# ── token minting / verification (HS256) ───────────────────────────────────
def mint_token(user_id: str, *, email: str | None = None,
               email_verified: bool = True, ttl_s: int = _TOKEN_TTL_S,
               now: int | None = None) -> str:
    """Sign a native session JWT. Claims mirror the Supabase shape (sub/email/
    aud/exp) so the rest of the stack treats native and Supabase tokens alike."""
    secret = native_secret()
    if not secret:
        raise RuntimeError("native auth secret not configured")
    iat = int(now if now is not None else time.time())
    claims = {
        "sub": str(user_id),
        "aud": _AUDIENCE,
        "iss": _ISSUER,
        "iat": iat,
        "exp": iat + int(ttl_s),
        "email": email or "",
        "email_verified": bool(email_verified),
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def verify_native_token(token: str, *, leeway: int = 60,
                        secret: str | None = None) -> dict:
    """Verify a native HS256 JWT → claims. Raises the shared typed AuthError."""
    sec = secret if secret is not None else native_secret()
    if not sec:
        raise TokenInvalid("native auth not configured")
    if not token or token.count(".") != 2:
        raise TokenInvalid("malformed token")
    try:
        return jwt.decode(
            token, sec, algorithms=["HS256"], audience=_AUDIENCE,
            leeway=leeway, options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired("token expired") from exc
    except jwt.InvalidAudienceError as exc:
        raise TokenInvalid("wrong audience") from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"invalid token: {exc}") from exc


_VERIFY_TTL_S = 60 * 60 * 24  # email-verification links expire in 24h


def mint_verification_token(user_id: str, *, ttl_s: int = _VERIFY_TTL_S,
                            now: int | None = None) -> str:
    """A single-purpose token for the email-verify link. Distinct from a session
    token: it carries ``purpose='email_verify'`` and NO ``aud``, so it can never
    be replayed as a login token (verify_native_token requires the audience)."""
    secret = native_secret()
    if not secret:
        raise RuntimeError("native auth secret not configured")
    iat = int(now if now is not None else time.time())
    claims = {"sub": str(user_id), "purpose": "email_verify",
              "iat": iat, "exp": iat + int(ttl_s)}
    return jwt.encode(claims, secret, algorithm="HS256")


def verify_verification_token(token: str, *, leeway: int = 60,
                              secret: str | None = None) -> dict:
    """Validate an email-verify token → claims. Raises typed ``AuthError``."""
    sec = secret if secret is not None else native_secret()
    if not sec:
        raise TokenInvalid("native auth not configured")
    if not token or token.count(".") != 2:
        raise TokenInvalid("malformed token")
    try:
        claims = jwt.decode(token, sec, algorithms=["HS256"], leeway=leeway,
                            options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired("verification link expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"invalid token: {exc}") from exc
    if claims.get("purpose") != "email_verify":
        raise TokenInvalid("not a verification token")
    return claims


_RESET_TTL_S = 60 * 30  # password-reset links expire in 30 minutes


def mint_reset_token(user_id: str, *, pw_hash: str | None = None,
                     ttl_s: int = _RESET_TTL_S, now: int | None = None) -> str:
    """A single-use-ish password-reset token. Binds to a short hash of the
    CURRENT password hash so the link auto-invalidates once the password
    changes (a used or superseded link stops working)."""
    secret = native_secret()
    if not secret:
        raise RuntimeError("native auth secret not configured")
    iat = int(now if now is not None else time.time())
    claims = {"sub": str(user_id), "purpose": "password_reset",
              "iat": iat, "exp": iat + int(ttl_s)}
    if pw_hash:
        claims["pwc"] = hashlib.sha256(pw_hash.encode()).hexdigest()[:16]
    return jwt.encode(claims, secret, algorithm="HS256")


def verify_reset_token(token: str, *, leeway: int = 60,
                       secret: str | None = None) -> dict:
    """Validate a password-reset token → claims. Raises typed ``AuthError``."""
    sec = secret if secret is not None else native_secret()
    if not sec:
        raise TokenInvalid("native auth not configured")
    if not token or token.count(".") != 2:
        raise TokenInvalid("malformed token")
    try:
        claims = jwt.decode(token, sec, algorithms=["HS256"], leeway=leeway,
                            options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpired("reset link expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"invalid token: {exc}") from exc
    if claims.get("purpose") != "password_reset":
        raise TokenInvalid("not a reset token")
    return claims


def pw_hash_fingerprint(pw_hash: str | None) -> str:
    return hashlib.sha256((pw_hash or "").encode()).hexdigest()[:16]


def is_native_token(token: str) -> bool:
    """Cheap unverified check: does the header say HS256? Used to route a token
    to the native verifier vs the Supabase JWKS one."""
    try:
        header = jwt.get_unverified_header(token)
        return str(header.get("alg", "")).upper() == "HS256"
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "AuthError", "TokenInvalid", "TokenExpired",
    "native_secret", "auth_mode", "native_enabled", "auth_enforced",
    "hash_password", "verify_password",
    "mint_token", "verify_native_token", "is_native_token",
]
