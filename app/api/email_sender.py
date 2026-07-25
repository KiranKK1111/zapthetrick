"""Transactional email over SMTP (vNext §10.1c — email verification).

Stdlib `smtplib` only (no new dependency), so it works with Gmail (an App
Password), Outlook, or any SMTP host. Config is env-first (loaded from `.env`
locally, or RunPod template env):

    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM (defaults to SMTP_USER), SMTP_STARTTLS (default "1"),
    SMTP_SSL (default "0" — set "1" for implicit-TLS port 465).

With no SMTP config, `email_enabled()` is False and the caller decides what to
do (registration refuses when verification is required). Sending is blocking, so
callers `await send_*` which offloads to a worker thread.
"""
from __future__ import annotations

import asyncio
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

_BRAND = "ZapTheTrick"
_ACCENT = "#8B5CF6"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def smtp_host() -> str:
    return _env("SMTP_HOST")


def smtp_from() -> str:
    return _env("SMTP_FROM") or _env("SMTP_USER")


def email_enabled() -> bool:
    """True when enough SMTP config is present to actually send."""
    return bool(smtp_host() and smtp_from())


def _smtp_settings() -> dict:
    return {
        "host": smtp_host(),
        "port": int(_env("SMTP_PORT", "587") or "587"),
        "user": _env("SMTP_USER"),
        "password": _env("SMTP_PASSWORD"),
        "from": smtp_from(),
        "starttls": _env("SMTP_STARTTLS", "1") not in ("0", "false", "no"),
        "ssl": _env("SMTP_SSL", "0") in ("1", "true", "yes"),
    }


def _send_sync(to: str, subject: str, html: str, text_alt: str) -> None:
    s = _smtp_settings()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((_BRAND, s["from"]))
    msg["To"] = to
    msg.set_content(text_alt)
    msg.add_alternative(html, subtype="html")

    if s["ssl"]:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(s["host"], s["port"], context=ctx, timeout=20) as srv:
            if s["user"]:
                srv.login(s["user"], s["password"])
            srv.send_message(msg)
        return
    with smtplib.SMTP(s["host"], s["port"], timeout=20) as srv:
        srv.ehlo()
        if s["starttls"]:
            srv.starttls(context=ssl.create_default_context())
            srv.ehlo()
        if s["user"]:
            srv.login(s["user"], s["password"])
        srv.send_message(msg)


async def send_email(to: str, subject: str, html: str, text_alt: str) -> None:
    """Send an email; raises on SMTP failure so the caller can surface it."""
    await asyncio.to_thread(_send_sync, to, subject, html, text_alt)


# ── verification email content ─────────────────────────────────────────────
def verification_subject() -> str:
    return f"Verify your {_BRAND} email"


def verification_html(verify_url: str, name: str | None = None) -> str:
    greeting = f"Hi {name}," if name else "Welcome!"
    # Table-based, fully-inlined CSS — the layout email clients actually honour.
    return f"""\
<!doctype html>
<html>
<body style="margin:0;padding:0;background:#0b0b0f;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b0b0f;padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
             style="max-width:480px;background:#15151c;border:1px solid #26262f;border-radius:16px;overflow:hidden;">
        <tr><td style="padding:32px 36px 8px;">
          <div style="font-size:20px;font-weight:700;color:#ffffff;letter-spacing:.2px;">
            <span style="color:{_ACCENT};">Zap</span>TheTrick
          </div>
        </td></tr>
        <tr><td style="padding:8px 36px 0;">
          <h1 style="margin:8px 0 4px;font-size:22px;line-height:1.3;color:#ffffff;font-weight:700;">
            Confirm your email
          </h1>
          <p style="margin:0 0 20px;font-size:15px;line-height:1.6;color:#a9adba;">
            {greeting} tap the button below to verify this address and activate
            your account. This link expires in 24 hours.
          </p>
        </td></tr>
        <tr><td align="center" style="padding:6px 36px 8px;">
          <a href="{verify_url}"
             style="display:inline-block;background:{_ACCENT};color:#ffffff;text-decoration:none;
                    font-size:15px;font-weight:600;padding:13px 28px;border-radius:10px;">
            Verify email
          </a>
        </td></tr>
        <tr><td style="padding:16px 36px 4px;">
          <p style="margin:0;font-size:12px;line-height:1.6;color:#6b6f7d;">
            Or paste this link into your browser:
          </p>
          <p style="margin:4px 0 0;font-size:12px;line-height:1.5;word-break:break-all;">
            <a href="{verify_url}" style="color:{_ACCENT};text-decoration:none;">{verify_url}</a>
          </p>
        </td></tr>
        <tr><td style="padding:24px 36px 32px;">
          <p style="margin:0;font-size:12px;line-height:1.6;color:#6b6f7d;border-top:1px solid #26262f;padding-top:16px;">
            If you didn't create a {_BRAND} account, you can safely ignore this email.
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def verification_text(verify_url: str) -> str:
    return (
        f"Confirm your {_BRAND} email\n\n"
        f"Verify this address by opening the link below (expires in 24 hours):\n"
        f"{verify_url}\n\n"
        f"If you didn't create a {_BRAND} account, you can ignore this email."
    )


async def send_verification_email(to: str, verify_url: str,
                                   name: str | None = None) -> None:
    await send_email(
        to,
        verification_subject(),
        verification_html(verify_url, name),
        verification_text(verify_url),
    )


# ── password-reset email ─────────────────────────────────────────────────────
def reset_html(reset_url: str, name: str | None = None) -> str:
    greeting = f"Hi {name}," if name else "Hi,"
    return f"""\
<!doctype html>
<html>
<body style="margin:0;padding:0;background:#0b0b0f;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b0b0f;padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
             style="max-width:480px;background:#15151c;border:1px solid #26262f;border-radius:16px;overflow:hidden;">
        <tr><td style="padding:32px 36px 8px;">
          <div style="font-size:20px;font-weight:700;color:#ffffff;">
            <span style="color:{_ACCENT};">Zap</span>TheTrick
          </div>
        </td></tr>
        <tr><td style="padding:8px 36px 0;">
          <h1 style="margin:8px 0 4px;font-size:22px;line-height:1.3;color:#ffffff;font-weight:700;">
            Reset your password
          </h1>
          <p style="margin:0 0 20px;font-size:15px;line-height:1.6;color:#a9adba;">
            {greeting} we got a request to reset your password. Tap below to
            choose a new one. This link expires in 30 minutes. If you didn't
            request this, you can ignore this email — your password won't change.
          </p>
        </td></tr>
        <tr><td align="center" style="padding:6px 36px 8px;">
          <a href="{reset_url}"
             style="display:inline-block;background:{_ACCENT};color:#ffffff;text-decoration:none;
                    font-size:15px;font-weight:600;padding:13px 28px;border-radius:10px;">
            Reset password
          </a>
        </td></tr>
        <tr><td style="padding:16px 36px 32px;">
          <p style="margin:0;font-size:12px;line-height:1.5;word-break:break-all;color:#6b6f7d;">
            Or paste this link: <a href="{reset_url}" style="color:{_ACCENT};text-decoration:none;">{reset_url}</a>
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


async def send_reset_email(to: str, reset_url: str,
                           name: str | None = None) -> None:
    await send_email(
        to, f"Reset your {_BRAND} password",
        reset_html(reset_url, name),
        f"Reset your {_BRAND} password (link expires in 30 min):\n{reset_url}",
    )


__all__ = [
    "email_enabled", "send_verification_email", "send_reset_email", "send_email",
    "verification_html", "verification_text", "verification_subject",
    "reset_html",
]
