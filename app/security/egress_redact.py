"""PII / secret redaction on the third-party-LLM egress boundary (§11, gap G13).

Free models are usually third-party endpoints that may log or train on inputs, so
raw content pasted into chat (API keys, credit cards, emails) must not leave the
device unredacted. This is the deterministic, always-on pass that §11 promised —
applied at the ONE egress choke point (the engine, before any provider call), so
every path (chat, triage, understanding, synthesis, retries) is covered.

Two levels (config `privacy.redact_egress`):
  * ``secrets`` (default) — redact only high-risk, low-utility tokens: API keys,
    cloud/provider tokens, private keys, credit cards, SSNs. These are almost
    never the legitimate subject of a question, so redacting them can't break a
    normal task while it closes the worst of the leak.
  * ``strict`` — also redact emails, phone numbers, and IPs (may reduce utility
    on turns whose subject IS an email/phone; opt-in).
  * ``off`` — no egress redaction.

Deterministic by design (a safety property, not model-decided) and fail-open:
any error returns the text unchanged — redaction must never break a call.
"""
from __future__ import annotations

import re

from app.agent_workspace.redact import redact_secrets

_PLACEHOLDER = "[REDACTED]"

# Credit-card-like (13–16 digits, optional separators) + SSN. Kept in "secrets"
# because they're always sensitive and never the legitimate subject of a turn.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Strict-mode PII.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .\-]?)?(?:\(?\d{2,4}\)?[ .\-]?){2,4}\d{2,4}(?!\w)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _redact_cards_ssn(text: str) -> str:
    def _card(m: re.Match) -> str:
        # only treat as a card when it has 13-16 digits (ignore separators)
        digits = re.sub(r"\D", "", m.group(0))
        return _PLACEHOLDER if 13 <= len(digits) <= 16 else m.group(0)
    text = _CARD_RE.sub(_card, text)
    return _SSN_RE.sub(_PLACEHOLDER, text)


# Bound every regex pass: very large messages are redacted in line-aligned
# segments so one pathological string can't hold the egress path hostage (the
# PEM pattern is DOTALL — a BEGIN with no END scans to end-of-string, which
# goes superlinear on unbounded input). A secret straddling a segment boundary
# is vanishingly rare (segments split at line breaks; only multi-line PEM
# bodies could straddle) and this is defense-in-depth, not the only guard.
_SEGMENT_CHARS = 262_144


def _segments(text: str):
    if len(text) <= _SEGMENT_CHARS:
        yield text
        return
    start, n = 0, len(text)
    while start < n:
        end = min(start + _SEGMENT_CHARS, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        yield text[start:end]
        start = end


def _redact_one(seg: str, mode: str) -> str:
    out = redact_secrets(seg)               # PEM/API/token/URL-cred/assignments
    out = _redact_cards_ssn(out)
    if mode == "strict":
        out = _EMAIL_RE.sub(_PLACEHOLDER, out)
        out = _IP_RE.sub(_PLACEHOLDER, out)
        out = _PHONE_RE.sub(_PLACEHOLDER, out)
    return out


def redact_text(text: str, *, mode: str = "secrets") -> tuple[str, int]:
    """Redact a single string. Returns (redacted, n_changes). Never raises."""
    if not text or mode == "off":
        return text, 0
    try:
        out = "".join(_redact_one(seg, mode) for seg in _segments(text))
        return out, (1 if out != text else 0)
    except Exception:  # noqa: BLE001 — redaction must never break a call
        return text, 0


def _redact_content(content, mode: str) -> tuple[object, int]:
    """Redact a message `content` — a string, or an OpenAI multipart list (only
    the text segments; image_url/base64 parts are left untouched)."""
    if isinstance(content, str):
        return redact_text(content, mode=mode)
    if isinstance(content, list):
        changed = 0
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                red, n = redact_text(part["text"], mode=mode)
                changed += n
                out.append({**part, "text": red})
            else:
                out.append(part)
        return out, changed
    return content, 0


def mode() -> str:
    try:
        from app.core.config_loader import cfg
        m = str(getattr(cfg.privacy, "redact_egress", "secrets")).strip().lower()
        return m if m in ("off", "secrets", "strict") else "secrets"
    except Exception:  # noqa: BLE001
        return "secrets"


def redact_messages(messages: list[dict], *, mode: str | None = None) -> list[dict]:
    """Return a copy of `messages` with PII/secrets redacted per the egress mode.
    A no-op (returns the input list) when the mode is 'off'. Never raises."""
    m = mode or globals()["mode"]()
    if m == "off" or not messages:
        return messages
    try:
        out = []
        for msg in messages:
            content = msg.get("content")
            red, n = _redact_content(content, m)
            out.append({**msg, "content": red} if n else msg)
        return out
    except Exception:  # noqa: BLE001
        return messages


__all__ = ["redact_text", "redact_messages", "mode"]
