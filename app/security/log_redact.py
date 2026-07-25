"""Redact secrets from log lines by construction (vNext §11.1).

The egress redactor (``egress_redact``) already scrubs content leaving the device
for a provider. §11.1 asks for the same guarantee on the OTHER sink that leaks —
logs and traces: an ``Authorization`` header, a JWT, or an API-key fragment must
not be able to reach a log line *because the sink refuses it*, not because every
call site remembered to scrub. This installs a redacting ``logging.Filter`` on
the log handlers, so it covers every logger that propagates to them — including
lines emitted by third-party libraries we don't control.

Attach to HANDLERS (not loggers): a filter on a logger only sees records logged
directly to it, while a filter on a handler sees every record that reaches the
handler (propagated children included). Fail-open: a redaction slip never drops
a log line.
"""
from __future__ import annotations

import logging

from app.security.egress_redact import redact_text


class RedactingFilter(logging.Filter):
    """Redacts secrets (API keys, tokens, JWTs, PEM keys, URL creds, ``k=secret``
    assignments) from the fully-interpolated message before any handler emits
    it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            red, n = redact_text(msg, mode="secrets")
            if n:
                # Replace the interpolated message and drop args so re-formatting
                # can't reintroduce the secret from record.args.
                record.msg = red
                record.args = ()
        except Exception:  # noqa: BLE001 — never let redaction drop a log line
            pass
        return True


# One shared instance — identity is how we detect an already-wrapped handler.
_FILTER = RedactingFilter(name="secret-redactor")

# The loggers whose handlers actually emit to the console in this app: the root
# (app loggers + basicConfig) and uvicorn's own logger tree.
_TARGET_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


def _has_redactor(handler: logging.Handler) -> bool:
    return any(isinstance(f, RedactingFilter) for f in handler.filters)


def install_log_redaction() -> int:
    """Attach the redactor to every current handler on the root + uvicorn
    loggers. Idempotent (safe to call repeatedly). Returns how many handlers were
    newly wrapped."""
    wrapped = 0
    seen: set[int] = set()
    for name in _TARGET_LOGGERS:
        lg = logging.getLogger(name) if name else logging.getLogger()
        for h in list(getattr(lg, "handlers", []) or []):
            if id(h) in seen:
                continue
            seen.add(id(h))
            if not _has_redactor(h):
                h.addFilter(_FILTER)
                wrapped += 1
    return wrapped


__all__ = ["RedactingFilter", "install_log_redaction"]
