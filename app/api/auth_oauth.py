"""Google OAuth for self-hosted accounts (vNext §10.1c).

Desktop loopback flow (no Supabase, no hosted callback): the FE opens Google's
consent page with a `http://127.0.0.1:<port>` redirect (a Google *Desktop* OAuth
client allows any loopback port), captures the `code`, and posts it here. We
exchange the code for an `id_token`, verify it against Google's public keys,
then upsert the user and mint one of OUR native HS256 JWTs — so a Google user
and an email/password user are the same kind of session downstream.

Config (env-first, like the auth secret): `GOOGLE_OAUTH_CLIENT_ID` +
`GOOGLE_OAUTH_CLIENT_SECRET`. Absent → Google sign-in is simply unavailable
(`/status` reports `google:false`) and the button stays disabled.
"""
from __future__ import annotations

import os
from urllib.parse import urlencode

import httpx
import jwt

from app.api.auth_jwt import TokenInvalid

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
_SCOPE = "openid email profile"


def _cfg_google():
    try:
        from app.core.config_loader import cfg
        a = getattr(cfg, "auth", None)
        return getattr(a, "google", None) if a is not None else None
    except Exception:  # noqa: BLE001
        return None


def google_client_id() -> str:
    env = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
    if env:
        return env
    g = _cfg_google()
    return str(getattr(g, "client_id", "") or "") if g is not None else ""


def google_client_secret() -> str:
    env = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
    if env:
        return env
    g = _cfg_google()
    return str(getattr(g, "client_secret", "") or "") if g is not None else ""


def google_enabled() -> bool:
    return bool(google_client_id() and google_client_secret())


def build_auth_url(redirect_uri: str, state: str) -> str:
    """The Google consent URL the FE opens in the system browser."""
    params = {
        "client_id": google_client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _SCOPE,
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
        "include_granted_scopes": "true",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


# Injectable so exchange_code can be tested without hitting Google.
_verify_hook = None
_jwk_client = None


def _verify_google_id_token(id_token: str) -> dict:
    if _verify_hook is not None:
        return _verify_hook(id_token)
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = jwt.PyJWKClient(GOOGLE_CERTS_URL)
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=google_client_id(),
            # Tolerate clock skew between this host and Google (else a host clock
            # a few seconds behind rejects a fresh token as "not yet valid (iat)").
            leeway=120,
            options={"require": ["exp", "sub", "email"]},
        )
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"bad google id_token: {exc}") from exc
    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise TokenInvalid("unexpected id_token issuer")
    return claims


class GoogleIdentity:
    def __init__(self, sub: str, email: str, name: str | None,
                 email_verified: bool):
        self.sub = sub
        self.email = email
        self.name = name
        self.email_verified = email_verified


async def exchange_code(code: str, redirect_uri: str) -> GoogleIdentity:
    """Trade the auth code for tokens, verify the id_token, return the identity."""
    data = {
        "code": code,
        "client_id": google_client_id(),
        "client_secret": google_client_secret(),
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(GOOGLE_TOKEN_URL, data=data)
    if r.status_code != 200:
        raise TokenInvalid(f"google token exchange failed ({r.status_code})")
    payload = r.json()
    id_token = payload.get("id_token")
    if not id_token:
        raise TokenInvalid("no id_token in google response")
    claims = _verify_google_id_token(id_token)
    return GoogleIdentity(
        sub=str(claims.get("sub")),
        email=str(claims.get("email") or "").lower(),
        name=claims.get("name"),
        email_verified=bool(claims.get("email_verified", False)),
    )


__all__ = [
    "google_enabled", "google_client_id", "build_auth_url", "exchange_code",
    "GoogleIdentity",
]
