"""Native account endpoints (vNext §10.1c) — the backend as identity provider.

  POST /api/auth/register   {email, password, name?}  -> {token, user}
  POST /api/auth/login      {email, password}          -> {token, user}
  GET  /api/auth/me                                    -> {user}         (auth)
  POST /api/auth/logout                                -> {ok}           (auth)
  GET  /api/auth/status                                -> {enabled, ...}

register/login/status are exempt from the auth gate (see auth.EXEMPT_PATHS) — you
can't present a token before you have one. me/logout require a valid token.

Self-hosted + dependency-free (PBKDF2 + HS256, `auth_native`), so accounts work
identically in a local desktop backend and on a RunPod pod. Enabled only when a
signing secret is configured (`ZAPTHETRICK_AUTH_SECRET`); otherwise every route
here returns 404 and the app stays in device-identity mode.
"""
from __future__ import annotations

import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import auth as _auth
from app.api import auth_native, auth_oauth, email_sender, rate_limit
from app.api.auth_jwt import AuthError
from storage.db import get_session
from storage.models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── schemas ────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=6, max_length=256)
    name: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., min_length=1, max_length=256)


class AuthUserOut(BaseModel):
    id: str
    email: str | None = None
    name: str | None = None
    email_verified: bool = True


class ProfileOut(BaseModel):
    id: str
    email: str | None = None
    full_name: str = ""
    # What the assistant should call the user (falls back to the full name's
    # first word when blank — see `personalization.profile.preferred_name`).
    display_name: str = ""
    # A small image as a data URL; "" when unset. See `personalization/profile.py`
    # for why it lives in `preferences` rather than blob storage.
    avatar: str = ""
    limits: dict = {}
    # Fields that were sent but could not be stored (e.g. an oversized avatar).
    # Reported rather than silently dropped.
    rejected: list[str] = []


class ProfileUpdate(BaseModel):
    """Every field optional: omitted = leave alone, "" = clear."""
    full_name: str | None = Field(default=None, max_length=200)
    display_name: str | None = Field(default=None, max_length=200)
    avatar: str | None = Field(default=None, max_length=300_000)


class AuthTokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUserOut


class RegisterResponse(BaseModel):
    # "verification_sent" → a verify email was sent, no token yet (the FE polls
    # /verify-status then logs in). "active" → verification off, token included.
    status: str
    email: str
    token: str | None = None
    user: AuthUserOut | None = None


class VerifyStatusResponse(BaseModel):
    exists: bool
    verified: bool


class AuthStatusResponse(BaseModel):
    enabled: bool          # any auth plane configured
    enforced: bool         # a real login wall (no token → rejected)
    mode: str              # native | supabase | off
    google: bool           # is OAuth Google available
    requires_verification: bool


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., max_length=320)


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(..., min_length=6, max_length=256)


class GoogleStartResponse(BaseModel):
    url: str
    state: str


class GoogleExchangeRequest(BaseModel):
    code: str
    redirect_uri: str
    state: str | None = None


def _require_native() -> None:
    if not auth_native.native_enabled():
        # Native endpoints are inert unless this backend hosts its own accounts.
        raise HTTPException(status_code=404, detail="Native auth is not enabled.")


def _client_ip(request: Request) -> str:
    # Honour a single proxy hop (RunPod fronts the pod), else the socket peer.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_guard(request: Request, action: str, ident: str,
                *, max_attempts: int, window_s: float) -> None:
    key = f"{action}:{_client_ip(request)}:{ident.lower()}"
    if not rate_limit.check_rate(key, max_attempts=max_attempts, window_s=window_s):
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait a minute and try again.")


def _user_out(u: User) -> AuthUserOut:
    return AuthUserOut(
        id=str(u.id), email=u.email, name=u.name,
        email_verified=bool(u.email_verified))


def _token_response(u: User) -> AuthTokenResponse:
    token = auth_native.mint_token(
        str(u.id), email=u.email, email_verified=bool(u.email_verified))
    return AuthTokenResponse(
        token=token, expires_in=auth_native._TOKEN_TTL_S, user=_user_out(u))


# ── endpoints ──────────────────────────────────────────────────────────────
@router.get("/status", response_model=AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """Let the client discover whether to show a login wall (enforced), offer
    optional accounts (enabled, not enforced), or hide auth entirely (off)."""
    mode = auth_native.auth_mode()
    return AuthStatusResponse(
        enabled=_auth.auth_active(),
        enforced=_auth.auth_enforced(),
        mode=mode,
        google=auth_oauth.google_enabled(),
        # Native: email verification is required when turned on AND SMTP is set.
        requires_verification=(
            mode == "native"
            and auth_native.require_email_verification()
            and email_sender.email_enabled()
        ),
    )


async def _send_verify(request: Request, user: User) -> None:
    """Mint a verify token and email the link to the user."""
    token = auth_native.mint_verification_token(str(user.id))
    base = auth_native.public_url() or str(request.base_url).rstrip("/")
    verify_url = f"{base}/api/auth/verify?token={token}"
    await email_sender.send_verification_email(user.email, verify_url, user.name)


@router.post("/register", response_model=RegisterResponse)
async def register(
    req: RegisterRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> RegisterResponse:
    _require_native()
    email = req.email.strip().lower()
    _rate_guard(request, "register", email, max_attempts=6, window_s=600)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="Enter a valid email.")
    if (pw_msg := auth_native.password_problem(req.password)):
        raise HTTPException(status_code=422, detail=pw_msg)

    require_verify = auth_native.require_email_verification()
    if require_verify and not email_sender.email_enabled():
        raise HTTPException(
            status_code=503,
            detail="Email verification is required but the email sender isn't "
                   "configured. Set SMTP_* in the backend environment.")

    existing = (
        await session.execute(
            select(User).where(func.lower(User.email) == email)
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.email_verified:
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists.")
        # Un-verified re-registration: refresh the password + re-send the link,
        # so a user who mistyped or lost the email can simply try again.
        existing.password_hash = auth_native.hash_password(req.password)
        if req.name:
            existing.name = req.name
        await session.commit()
        await session.refresh(existing)
        await _send_verify(request, existing)
        return RegisterResponse(status="verification_sent", email=email)

    user = User(
        email=email,
        name=(req.name or None),
        password_hash=auth_native.hash_password(req.password),
        email_verified=(not require_verify),
        preferences={"account": "native"},
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    if require_verify:
        await _send_verify(request, user)
        return RegisterResponse(status="verification_sent", email=email)
    # Verification disabled → the account is active immediately (token in-band).
    tok = _token_response(user)
    return RegisterResponse(
        status="active", email=email, token=tok.token, user=tok.user)


@router.post("/login", response_model=AuthTokenResponse)
async def login(
    req: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> AuthTokenResponse:
    _require_native()
    email = req.email.strip().lower()
    _rate_guard(request, "login", email, max_attempts=10, window_s=300)
    user = (
        await session.execute(
            select(User).where(func.lower(User.email) == email)
        )
    ).scalar_one_or_none()
    # Verify even when the user is missing (dummy hash) to blunt user enumeration
    # via timing, then fail with one generic message.
    ok = auth_native.verify_password(
        req.password, user.password_hash if user else None)
    if user is None or not ok:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if auth_native.require_email_verification() and not user.email_verified:
        # 403 so the FE can distinguish "wrong password" (401) from "verify first".
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before signing in.")
    return _token_response(user)


@router.get("/me", response_model=AuthUserOut)
async def me(session: AsyncSession = Depends(get_session)) -> AuthUserOut:
    uid = _auth.current_user_id()
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        user = await session.get(User, uuid.UUID(str(uid)))
    except (ValueError, TypeError):
        user = None
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    return _user_out(user)


async def _current_user(session: AsyncSession) -> User:
    """The signed-in user, or 401. Shared by the profile routes below."""
    uid = _auth.current_user_id()
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        user = await session.get(User, uuid.UUID(str(uid)))
    except (ValueError, TypeError):
        user = None
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    return user


@router.get("/profile", response_model=ProfileOut)
async def get_profile(
    session: AsyncSession = Depends(get_session),
) -> ProfileOut:
    """The editable profile behind the Profile screen.

    Separate from `/me` on purpose: `/me` is the auth identity (id, email,
    verification state) that the sign-in flow depends on, and widening it would
    make every auth response carry an avatar. This is the presentation profile.
    """
    from app.personalization.profile import profile_payload
    return ProfileOut(**profile_payload(await _current_user(session)))


@router.patch("/profile", response_model=ProfileOut)
async def update_profile(
    body: ProfileUpdate,
    session: AsyncSession = Depends(get_session),
) -> ProfileOut:
    """Update the profile. Every field is OPTIONAL and only applied when present.

    Omitting a field leaves it untouched; sending an explicit empty string CLEARS
    it. That distinction matters for the avatar and the preferred name, where
    "don't change this" and "remove this" are different intents — a PUT-style
    whole-object write would have made clearing indistinguishable from a client
    that simply didn't load the field yet.

    The full name writes through to `User.name` (a real column, used by auth); the
    preferred name and avatar go into `User.preferences`. An avatar that fails
    validation is reported in `rejected` and the rest of the profile still saves —
    a bad image must not block a name change.
    """
    from app.personalization.profile import (
        clean_full_name, clean_avatar, profile_payload, set_avatar,
        set_display_name,
    )

    user = await _current_user(session)
    rejected: list[str] = []

    if body.full_name is not None:
        user.name = clean_full_name(body.full_name) or None
    if body.display_name is not None:
        user.preferences = set_display_name(user.preferences, body.display_name)
    if body.avatar is not None:
        raw = body.avatar.strip()
        if raw and not clean_avatar(raw):
            # Present but unusable — say so instead of silently dropping it.
            rejected.append(
                "avatar must be a PNG/JPEG/WebP/GIF data URL under "
                "192 KB (SVG is not accepted)")
        else:
            user.preferences = set_avatar(user.preferences, raw)

    await session.commit()
    await session.refresh(user)
    return ProfileOut(**profile_payload(user), rejected=rejected)


@router.delete("/account")
async def delete_account(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """PERMANENTLY delete the current user's account and everything keyed to
    it. Two passes: `data_lifecycle.delete_all` sweeps conversations, memory,
    projects, vectors, and blobs (the same sweep the privacy wipe uses), then
    the User row itself is deleted — resumes, API keys, and usage rows go with
    it via their `ondelete=CASCADE` foreign keys. Irreversible by design."""
    uid = _auth.current_user_id()
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        user = await session.get(User, uuid.UUID(str(uid)))
    except (ValueError, TypeError):
        user = None
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    try:
        from app.memory import data_lifecycle
        await data_lifecycle.delete_all(session, user_id=str(user.id))
    except Exception:  # noqa: BLE001 — the user-row cascade below still wipes
        pass           # relational data even if the vector/blob sweep failed.
    await session.delete(user)
    await session.commit()
    return {"deleted": True}


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(
    session: AsyncSession = Depends(get_session),
) -> AuthTokenResponse:
    """Re-issue a fresh token for the CURRENT (still-valid) session so an active
    user's 30-day session doesn't lapse under them. Requires a valid token — the
    middleware already verified it; an expired/invalid one → 401 → re-login."""
    _require_native()
    uid = _auth.current_user_id()
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        user = await session.get(User, uuid.UUID(str(uid)))
    except (ValueError, TypeError):
        user = None
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    return _token_response(user)


def _verify_result_page(ok: bool, message: str) -> str:
    accent = "#8B5CF6"
    icon = "✓" if ok else "✕"
    ring = accent if ok else "#e5484d"
    title = "Email verified" if ok else "Verification failed"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title></head>
<body style="margin:0;background:#0b0b0f;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;">
  <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;">
    <div style="max-width:420px;text-align:center;background:#15151c;border:1px solid #26262f;
                border-radius:16px;padding:40px 32px;">
      <div style="width:64px;height:64px;border-radius:50%;margin:0 auto 20px;
                  display:flex;align-items:center;justify-content:center;
                  background:{ring}1a;color:{ring};font-size:32px;">{icon}</div>
      <h1 style="margin:0 0 8px;color:#fff;font-size:22px;">{title}</h1>
      <p style="margin:0;color:#a9adba;font-size:15px;line-height:1.6;">{message}</p>
    </div>
  </div></body></html>"""


@router.get("/verify", response_class=HTMLResponse)
async def verify_email(
    token: str,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """The email link target. Verifies the token, marks the account confirmed,
    and shows a branded page telling the user to return to the app."""
    try:
        claims = auth_native.verify_verification_token(token)
    except auth_native.TokenExpired:
        return HTMLResponse(_verify_result_page(
            False, "This link has expired. Sign up again to get a new one."),
            status_code=400)
    except AuthError:
        return HTMLResponse(_verify_result_page(
            False, "This verification link is invalid."), status_code=400)
    try:
        user = await session.get(User, uuid.UUID(str(claims.get("sub"))))
    except (ValueError, TypeError):
        user = None
    if user is None:
        return HTMLResponse(_verify_result_page(
            False, "We couldn't find that account."), status_code=404)
    if not user.email_verified:
        user.email_verified = True
        await session.commit()
    return HTMLResponse(_verify_result_page(
        True, "Your email is confirmed. Return to the ZapTheTrick app — "
              "you'll be signed in automatically."))


@router.get("/verify-status", response_model=VerifyStatusResponse)
async def verify_status(
    email: str,
    session: AsyncSession = Depends(get_session),
) -> VerifyStatusResponse:
    """Polled by the app after registration: has this email been verified yet?"""
    e = email.strip().lower()
    user = (
        await session.execute(select(User).where(func.lower(User.email) == e))
    ).scalar_one_or_none()
    return VerifyStatusResponse(
        exists=user is not None, verified=bool(user and user.email_verified))


@router.post("/logout")
async def logout() -> dict:
    # Stateless JWTs — the client discards the token (and calls AuthSession.clear).
    # A future token-blocklist would plug in here for server-side revocation.
    return {"ok": True}


# ── Password reset ───────────────────────────────────────────────────────────
@router.post("/forgot-password")
async def forgot_password(
    req: ForgotPasswordRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Email a password-reset link. Always returns ok (never reveals whether an
    email is registered) — but only actually sends for a real native account."""
    _require_native()
    email = req.email.strip().lower()
    _rate_guard(request, "forgot", email, max_attempts=4, window_s=900)
    user = (
        await session.execute(select(User).where(func.lower(User.email) == email))
    ).scalar_one_or_none()
    if user is not None and user.password_hash and email_sender.email_enabled():
        token = auth_native.mint_reset_token(
            str(user.id), pw_hash=user.password_hash)
        base = auth_native.public_url() or str(request.base_url).rstrip("/")
        reset_url = f"{base}/api/auth/reset-password?token={token}"
        try:
            await email_sender.send_reset_email(user.email, reset_url, user.name)
        except Exception:  # noqa: BLE001 — don't leak delivery failures/timing
            pass
    return {"ok": True}


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(token: str) -> HTMLResponse:
    """The link target — a minimal branded page that POSTs a new password back."""
    try:
        auth_native.verify_reset_token(token)
    except auth_native.TokenExpired:
        return HTMLResponse(_verify_result_page(
            False, "This reset link has expired. Request a new one."),
            status_code=400)
    except AuthError:
        return HTMLResponse(_verify_result_page(
            False, "This reset link is invalid."), status_code=400)
    return HTMLResponse(_reset_form_page(token))


@router.post("/reset-password")
async def reset_password(
    req: ResetPasswordRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    _require_native()
    if (pw_msg := auth_native.password_problem(req.password)):
        raise HTTPException(status_code=422, detail=pw_msg)
    try:
        claims = auth_native.verify_reset_token(req.token)
    except auth_native.TokenExpired:
        raise HTTPException(status_code=400, detail="This reset link has expired.")
    except AuthError:
        raise HTTPException(status_code=400, detail="Invalid reset link.")
    try:
        user = await session.get(User, uuid.UUID(str(claims.get("sub"))))
    except (ValueError, TypeError):
        user = None
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")
    # If the token was bound to the old password hash, reject once it's changed.
    pwc = claims.get("pwc")
    if pwc and pwc != auth_native.pw_hash_fingerprint(user.password_hash):
        raise HTTPException(status_code=400, detail="This reset link was already used.")
    user.password_hash = auth_native.hash_password(req.password)
    user.email_verified = True  # completing a reset proves email control
    await session.commit()
    return {"ok": True}


def _reset_form_page(token: str) -> str:
    accent = "#8B5CF6"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reset password</title></head>
<body style="margin:0;background:#0b0b0f;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;">
  <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;">
    <div style="max-width:380px;width:100%;background:#15151c;border:1px solid #26262f;
                border-radius:16px;padding:32px;">
      <h1 style="margin:0 0 6px;color:#fff;font-size:20px;">Choose a new password</h1>
      <p style="margin:0 0 20px;color:#a9adba;font-size:14px;">At least 6 characters.</p>
      <input id="pw" type="password" placeholder="New password"
             style="width:100%;box-sizing:border-box;padding:12px;border-radius:10px;
                    border:1px solid #33333d;background:#0f0f15;color:#fff;font-size:15px;">
      <div id="msg" style="color:#e5484d;font-size:13px;min-height:18px;margin:8px 2px;"></div>
      <button id="btn" style="width:100%;background:{accent};color:#fff;border:0;
              padding:13px;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;">
        Reset password
      </button>
    </div>
  </div>
  <script>
    const btn=document.getElementById('btn'),pw=document.getElementById('pw'),msg=document.getElementById('msg');
    btn.onclick=async()=>{{
      if((pw.value||'').length<6){{msg.textContent='Password must be at least 6 characters.';return;}}
      btn.disabled=true;btn.textContent='Resetting…';
      try{{
        const r=await fetch('/api/auth/reset-password',{{method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{token:{token!r},password:pw.value}})}});
        if(r.ok){{document.body.innerHTML='<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;color:#fff;font-family:sans-serif;text-align:center;padding:24px"><div><h2>Password updated ✓</h2><p style=\"color:#a9adba\">Return to the app and sign in with your new password.</p></div></div>';}}
        else{{const j=await r.json().catch(()=>({{}}));msg.textContent=j.detail||'Reset failed.';btn.disabled=false;btn.textContent='Reset password';}}
      }}catch(e){{msg.textContent='Network error.';btn.disabled=false;btn.textContent='Reset password';}}
    }};
    pw.addEventListener('keydown',e=>{{if(e.key==='Enter')btn.click();}});
  </script>
</body></html>"""


# ── Google OAuth (desktop loopback) ─────────────────────────────────────────
@router.get("/google/start", response_model=GoogleStartResponse)
async def google_start(redirect_uri: str) -> GoogleStartResponse:
    """Return the Google consent URL for the FE to open. `redirect_uri` is the
    FE's ephemeral loopback (`http://127.0.0.1:<port>`)."""
    if not auth_oauth.google_enabled():
        raise HTTPException(status_code=404, detail="Google sign-in is not configured.")
    state = secrets.token_urlsafe(24)
    return GoogleStartResponse(
        url=auth_oauth.build_auth_url(redirect_uri, state), state=state)


@router.post("/google/exchange", response_model=AuthTokenResponse)
async def google_exchange(
    req: GoogleExchangeRequest,
    session: AsyncSession = Depends(get_session),
) -> AuthTokenResponse:
    """Trade the loopback `code` for a Google identity, upsert the user, and
    mint one of our native session tokens."""
    _require_native()  # we still mint OUR HS256 token, so a secret is required
    if not auth_oauth.google_enabled():
        raise HTTPException(status_code=404, detail="Google sign-in is not configured.")
    try:
        identity = await auth_oauth.exchange_code(req.code, req.redirect_uri)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=f"Google sign-in failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Google sign-in error: {exc}")
    if not identity.email:
        raise HTTPException(status_code=401, detail="Google account has no email.")
    user = await _upsert_oauth_user(session, identity)
    return _token_response(user)


async def _upsert_oauth_user(
        session: AsyncSession, identity: auth_oauth.GoogleIdentity) -> User:
    email = identity.email.lower()
    user = (
        await session.execute(select(User).where(func.lower(User.email) == email))
    ).scalar_one_or_none()
    if user is None:
        user = User(
            email=email,
            name=identity.name,
            password_hash=None,  # OAuth account — no local password
            email_verified=bool(identity.email_verified),
            preferences={"account": "google", "oauth_sub": identity.sub},
        )
        session.add(user)
    else:
        # Keep the profile fresh; a Google login also verifies the email.
        if identity.name and not user.name:
            user.name = identity.name
        if identity.email_verified:
            user.email_verified = True
    await session.commit()
    await session.refresh(user)
    return user
