"""Auth middleware + per-request user context (vNext §10.1b / §10.2).

Injects the authenticated ``user_id`` ONCE, at the edge, so nothing downstream
re-parses a token. REST/SSE read ``Authorization: Bearer <jwt>``; the WS handshake
authenticates before any frame is processed (see ``authenticate_ws_first_frame``).

Flag-gated by ``auth.enabled`` (default OFF): with auth off this is a pass-through
and routes keep resolving the device user exactly as today. With auth ON, every
route except a small health/allow-list requires a valid JWT and gets a typed 401
so a modified client can't skip verification. Fail-CLOSED.

A pure ASGI middleware (not ``BaseHTTPMiddleware``) so the ``user_id`` ContextVar
is set in the SAME task the route runs in — reliable contextvar propagation.
"""
from __future__ import annotations

import uuid

from starlette.responses import JSONResponse

from app.api import auth_jwt, auth_native
# The user-id ContextVar lives in the storage layer (neutral) so the device-user
# resolvers can read the SAME value the middleware sets here — that's what makes
# every route scope to the logged-in user without per-call-site edits.
from storage.context import current_user_id_var as _current_user_id

# Paths served WITHOUT auth: the pod's front door for connecting + probing, plus
# the public auth endpoints (you can't hold a token before you log in).
# (`/healthz` is the liveness probe the watchdog/proxy hit; `/readyz` is the
# connect-screen probe.)
EXEMPT_PATHS = frozenset({
    "/healthz", "/readyz", "/", "/docs", "/openapi.json", "/redoc",
    # `/api/health` is the connect-screen reachability probe — it must work
    # BEFORE sign-in (you can't authenticate against a server you haven't
    # connected to yet), so it's exempt. The /api/health/* SUB-routes
    # (dashboard, crashes, self-heal…) are NOT exempt and still require auth.
    "/api/health",
    "/api/models/warmup",
    "/api/auth/register", "/api/auth/login", "/api/auth/status",
    "/api/auth/google/start", "/api/auth/google/exchange",
    "/api/auth/verify", "/api/auth/verify-status",
    "/api/auth/forgot-password", "/api/auth/reset-password",
    # Blobs auth via a ?token= query param (Image.network can't send a header),
    # and enforce ownership inside the route — so they're exempt from the header
    # gate but NOT unauthenticated.
    "/api/blob", "/api/blob/preview",
})


def current_user_id() -> str | None:
    """The verified Supabase user id for the current request (None if auth off)."""
    return _current_user_id.get()


def _is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith("/docs")


def bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def auth_active() -> bool:
    """Any auth plane configured (native accounts OR a Supabase project). When
    false the middleware is a pass-through — device-identity, byte-identical."""
    return auth_native.native_enabled() or auth_jwt.auth_enabled()


def auth_enforced() -> bool:
    """Whether a missing/invalid token is REJECTED (a real login wall). Supabase
    mode always enforces; native mode enforces only when opted in (RunPod). When
    not enforced (local default), accounts are OPTIONAL: a valid token scopes the
    user, no token → anonymous device-identity, exactly like today."""
    return auth_jwt.auth_enabled() or auth_native.auth_enforced()


def verify_any_token(token: str, *, jwks=None) -> dict:
    """Verify a native HS256 token or a Supabase RS256 token — one entry point so
    the gate and the WS handshake treat both alike. Raises typed ``AuthError``."""
    if auth_native.is_native_token(token) and auth_native.native_secret():
        return auth_native.verify_native_token(token)
    return auth_jwt.verify_token(token, jwks=jwks)


def authorize(path: str, auth_header: str | None, *, jwks=None
              ) -> tuple[str | None, dict | None]:
    """Decide a request. Returns ``(user_id, error)``: an exempt/auth-off request
    or an anonymous request in optional-accounts mode returns ``(None, None)``
    (allow, no auth user); a rejection returns ``(None, {status,code,detail})``;
    a verified request returns ``(user_id, None)``."""
    if _is_exempt(path) or not auth_active():
        return None, None
    enforced = auth_enforced()
    token = bearer_token(auth_header)
    if not token:
        # ONLY a missing token respects the optional/anonymous mode. A MISSING
        # token in optional mode → anonymous (legit). Enforced → reject.
        if enforced:
            return None, {"status": 401, "code": "token_missing",
                          "detail": "Authentication required."}
        return None, None
    # A token IS present → always verify it, even in optional mode. Silently
    # degrading a bad/expired/unverified token to the device user makes a signed-
    # in client appear to work but see the WRONG (anonymous) data — the class of
    # bug where "I'm logged in but my chats are gone". Surface it so the client
    # re-authenticates.
    try:
        claims = verify_any_token(token, jwks=jwks)
        auth_jwt.enforce_email_verified(claims)
    except auth_jwt.EmailUnverified:
        return None, {"status": 403, "code": "email_unverified",
                      "detail": "Please verify your email to continue."}
    except auth_jwt.TokenExpired:
        return None, {"status": 401, "code": "token_expired",
                      "detail": "Your session expired — sign in again."}
    except auth_jwt.AuthError as exc:
        return None, {"status": 401, "code": getattr(exc, "code", "token_invalid"),
                      "detail": "Invalid credentials."}
    return auth_jwt.user_id_from_claims(claims), None


_device_user_cache = None


async def _cached_device_user():
    """The device user's UUID, resolved once and cached (the device install id is
    stable per deploy, so this is a one-time DB lookup). Used by the middleware to
    bind a real user in auth-off mode. Returns None on any failure (fail-open)."""
    global _device_user_cache
    if _device_user_cache is None:
        try:
            from storage.device import ensure_device_user
            _device_user_cache = await ensure_device_user()
        except Exception:  # noqa: BLE001 — never break the request over this
            return None
    return _device_user_cache


async def ws_user_id(uid) -> str | None:
    """The user id a WebSocket should bind into its context: the authenticated
    uid, or the DEVICE user in auth-off mode — so WS-path scoping (router/cache/
    catalog) matches HTTP and `resolve_user_id()` instead of running under None."""
    if uid:
        return str(uid)
    if not auth_enforced():
        dev = await _cached_device_user()
        return str(dev) if dev else None
    return None


class AuthMiddleware:
    """Pure ASGI auth gate. ``jwks`` overridable for tests."""

    def __init__(self, app, jwks=None):
        self.app = app
        self.jwks = jwks

    async def __call__(self, scope, receive, send):
        # Non-HTTP (WS/lifespan) and CORS preflight (OPTIONS carries no auth) pass
        # straight through — the WS handshake authenticates itself (§10.1b).
        if scope.get("type") != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        auth_header = None
        for k, v in scope.get("headers") or []:
            if k == b"authorization":
                auth_header = v.decode("latin-1")
                break
        user_id, error = authorize(path, auth_header, jwks=self.jwks)
        if error is not None:
            resp = JSONResponse(
                {"code": error["code"], "detail": error["detail"]},
                status_code=error["status"])
            await resp(scope, receive, send)
            return
        # Bind a user for the WHOLE request. When there's no authenticated user
        # (auth off / anonymous / device mode), fall back to the DEVICE user —
        # NOT None — so every scoped path (`get_request_user_id()` in catalog,
        # keys, discovery, cache, router) agrees with `resolve_user_id()`.
        # Leaving it None strands writes under user_id=NULL while reads look under
        # the device user → "keys healthy but 0 models". Enforced mode never
        # reaches here with user_id=None (authorize returns a 401 above).
        if user_id is None and not auth_enforced():
            _dev = await _cached_device_user()
            if _dev is not None:
                user_id = str(_dev)
        token = _current_user_id.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user_id.reset(token)


def authenticate_ws(token: str | None, *, jwks=None
                    ) -> tuple[str | None, dict | None]:
    """Authenticate a WebSocket via a ``?token=`` query param (the middleware
    only covers HTTP, so WS must authenticate itself). Returns ``(user_id,
    error)``. Not enforced → always allowed; a token present is still verified so
    the connection is scoped to that user, but a bad token degrades to anonymous.
    Enforced → a valid token is required (else a typed error to close the socket)."""
    enforced = auth_enforced()
    if token:
        # A present token is always verified (see authorize()): a bad token must
        # not silently scope the socket to the anonymous device user.
        try:
            claims = verify_any_token(token, jwks=jwks)
            auth_jwt.enforce_email_verified(claims)
            return auth_jwt.user_id_from_claims(claims), None
        except auth_jwt.AuthError as exc:
            return None, {"code": getattr(exc, "code", "token_invalid"),
                          "detail": "Invalid token."}
    # No token.
    if enforced:
        return None, {"code": "auth_required",
                      "detail": "Authentication required."}
    return None, None


def authenticate_ws_first_frame(msg, *, jwks=None) -> tuple[str | None, dict | None]:
    """Verify the FIRST WS frame (the handshake). When auth is on the frame must
    be ``{"type":"auth","token":"<jwt>"}`` — anything else closes the socket.
    When auth is off/optional → ``(None, None)`` (no auth required, today's
    behaviour); the socket only demands an auth frame when auth is enforced."""
    if not auth_enforced():
        return None, None
    if not isinstance(msg, dict) or msg.get("type") != "auth" or not msg.get("token"):
        return None, {"code": "auth_required",
                      "detail": "The first WS frame must be an auth frame."}
    try:
        claims = verify_any_token(str(msg["token"]), jwks=jwks)
        auth_jwt.enforce_email_verified(claims)
    except auth_jwt.AuthError as exc:
        return None, {"code": getattr(exc, "code", "token_invalid"),
                      "detail": "Invalid token."}
    return auth_jwt.user_id_from_claims(claims), None


async def resolve_user_id() -> uuid.UUID | None:
    """The current user's DB id — the unified resolver routes adopt in place of
    ``ensure_device_user()``. Auth ON → the JWT ``sub`` (provisioned as a users
    row in §10.2); auth OFF → the device user, i.e. byte-identical to today."""
    uid = current_user_id()
    if uid:
        try:
            return uuid.UUID(str(uid))
        except (ValueError, TypeError):
            return None
    from storage.device import ensure_device_user
    return await ensure_device_user()


__all__ = [
    "AuthMiddleware", "current_user_id", "authorize", "bearer_token",
    "authenticate_ws_first_frame", "authenticate_ws", "resolve_user_id",
    "EXEMPT_PATHS", "auth_active", "auth_enforced", "verify_any_token",
]
