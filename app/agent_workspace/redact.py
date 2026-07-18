"""Secret redaction for agent step traces (Phase 4.5).

Uploaded codebases routinely contain credentials — `.env` files, private keys,
cloud tokens, DB connection strings. The agent legitimately reads those files
to do its work, but their CONTENTS must never be echoed back into the streamed
/ persisted step trace (where they'd be visible in chat history, logs, or a
shared session). `redact_secrets` scrubs the common shapes, replacing the
secret value with `[REDACTED]` while keeping enough structure that the trace is
still readable ("API_KEY=[REDACTED]").

Conservative by design: it targets high-signal patterns (provider key formats,
PEM blocks, `key=value` secret assignments, URLs with inline credentials) so it
won't shred ordinary source code. Best-effort — defense in depth, not a
guarantee; it never raises.
"""
from __future__ import annotations

import re

_PLACEHOLDER = "[REDACTED]"

# PEM private-key blocks (RSA/EC/OPENSSH/PGP) — collapse the whole body.
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

# Provider-specific token formats (high precision — these shapes are secrets).
_TOKEN_RES = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                 # AWS access key id
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),                 # AWS temp key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),       # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),     # GitHub fine-grained
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),     # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),           # Google API key
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),              # OpenAI-style
    re.compile(r"\bsk-ant-[A-Za-z0-9\-]{20,}\b"),        # Anthropic-style
    re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"),        # NVIDIA
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),        # GitLab PAT
    # JWTs (header.payload.signature, all base64url).
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
]

# URLs carrying inline credentials: scheme://user:password@host → mask password.
_URL_CRED_RE = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:)([^\s@/]+)(@)"
)

# `key = value` / `key: value` assignments where the KEY names a secret.
# Captures an optional quote so we can preserve it around the placeholder.
_ASSIGN_RE = re.compile(
    r"""(?ix)
    \b(
        (?:[A-Z0-9_]*)?
        (?:secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|
           password|passwd|pwd|auth|private[_-]?key|credential|
           encryption[_-]?key|session[_-]?key)
        (?:[A-Z0-9_]*)?
    )
    (\s*[:=]\s*)
    (["']?)
    ([^\s"',;]{4,})
    (["']?)
    """,
)


def redact_secrets(text: str) -> str:
    """Return `text` with secret-looking substrings replaced by `[REDACTED]`.
    Never raises — returns the input unchanged on any internal error."""
    if not text:
        return text
    try:
        s = _PEM_RE.sub(_PLACEHOLDER, text)
        for rx in _TOKEN_RES:
            s = rx.sub(_PLACEHOLDER, s)
        s = _URL_CRED_RE.sub(rf"\1{_PLACEHOLDER}\3", s)

        def _assign(m: re.Match) -> str:
            key, sep, q, _val, q2 = m.groups()
            close = q or q2
            return f"{key}{sep}{q}{_PLACEHOLDER}{close}"

        s = _ASSIGN_RE.sub(_assign, s)
        return s
    except Exception:  # noqa: BLE001 — redaction must never break a run
        return text


def redact_event(evt: dict) -> dict:
    """Scrub secret values from the free-text fields of an agent SSE event,
    in place, returning it. Targets the fields that carry file/command output
    or model prose: `result`, `message`, `text`, `feedback`, `summary`,
    `detail`, and a tool call's string args."""
    if not isinstance(evt, dict):
        return evt
    for k in ("result", "message", "text", "feedback", "summary", "detail",
              "plan", "answer"):
        v = evt.get(k)
        if isinstance(v, str):
            evt[k] = redact_secrets(v)
    args = evt.get("args")
    if isinstance(args, dict):
        for ak, av in list(args.items()):
            if isinstance(av, str):
                args[ak] = redact_secrets(av)
    return evt


__all__ = ["redact_secrets", "redact_event"]
