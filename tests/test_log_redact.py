"""Secret redaction on the logging sink (vNext §11.1).

Pins: a secret in a log record is scrubbed before a handler can emit it, normal
lines pass through untouched, and installation is idempotent.
"""
from __future__ import annotations

import logging

from app.security.log_redact import RedactingFilter, install_log_redaction


def _record(msg, *args):
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, args, None)


def test_filter_redacts_secret_in_message():
    f = RedactingFilter()
    rec = _record("connecting with api_key=sk-abcdef0123456789abcdef0123456789")
    assert f.filter(rec) is True                 # never drops the line
    assert "sk-abcdef0123456789" not in rec.getMessage()
    assert "[REDACTED]" in rec.getMessage()


def test_filter_redacts_secret_supplied_via_args():
    # getMessage() interpolates args; redaction must catch a secret that only
    # appears after interpolation, and clear args so it can't reappear.
    f = RedactingFilter()
    secret = "api_key=sk-abcdef0123456789abcdef0123456789"
    rec = _record("connecting with %s", secret)
    f.filter(rec)
    out = rec.getMessage()
    assert "sk-abcdef0123456789" not in out
    assert "[REDACTED]" in out
    assert rec.args == ()


def test_filter_leaves_normal_lines_untouched():
    f = RedactingFilter()
    rec = _record("routing: picked model %s", "llama-3.3-70b")
    f.filter(rec)
    assert rec.getMessage() == "routing: picked model llama-3.3-70b"


def test_install_is_idempotent():
    logger = logging.getLogger("uvicorn")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        install_log_redaction()
        install_log_redaction()   # second call must not double-wrap
        redactors = [f for f in handler.filters
                     if isinstance(f, RedactingFilter)]
        assert len(redactors) == 1
    finally:
        logger.removeHandler(handler)
