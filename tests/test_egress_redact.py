"""PII/secret redaction on the LLM egress boundary (gap G13)."""
from __future__ import annotations

from app.security import egress_redact as er


def test_secrets_mode_redacts_keys_and_tokens():
    for secret in ("sk-abcdefghij1234567890ABCD",
                   "ghp_abcdefghijklmnopqrstuvwx1234",
                   "AKIAIOSFODNN7EXAMPLE",
                   "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456"):
        out, n = er.redact_text(f"here is {secret} ok", mode="secrets")
        assert "[REDACTED]" in out and secret not in out and n == 1


def test_secrets_mode_redacts_cards_and_ssn():
    assert "[REDACTED]" in er.redact_text("pay 4111 1111 1111 1111", mode="secrets")[0]
    assert "[REDACTED]" in er.redact_text("ssn 123-45-6789", mode="secrets")[0]


def test_secrets_mode_keeps_email_and_normal_text():
    out, n = er.redact_text("email bob@acme.com about the meeting", mode="secrets")
    assert out == "email bob@acme.com about the meeting" and n == 0


def test_strict_mode_redacts_pii():
    assert "[REDACTED]" in er.redact_text("bob@acme.com", mode="strict")[0]
    assert "[REDACTED]" in er.redact_text("server 10.0.0.5", mode="strict")[0]


def test_off_mode_is_noop():
    out, n = er.redact_text("sk-abcdefghij1234567890ABCD", mode="off")
    assert n == 0 and "sk-" in out


def test_redact_messages_covers_all_roles():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "token: ghp_abcdefghijklmnopqrstuvwx1234"},
    ]
    out = er.redact_messages(msgs, mode="secrets")
    assert out[0]["content"] == "you are helpful"       # unchanged
    assert "[REDACTED]" in out[1]["content"]


def test_redact_messages_multipart_leaves_image_untouched():
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "text", "text": "my key sk-abcdefghij1234567890ABCD"},
    ]}]
    out = er.redact_messages(msgs, mode="secrets")
    parts = out[0]["content"]
    assert parts[0]["type"] == "image_url"              # image untouched
    assert parts[0]["image_url"]["url"].endswith("AAA")
    assert "[REDACTED]" in parts[1]["text"]


def test_redact_messages_off_returns_input():
    msgs = [{"role": "user", "content": "sk-abcdefghij1234567890ABCD"}]
    assert er.redact_messages(msgs, mode="off") is msgs


def test_fail_open_on_bad_content():
    # non-str/list content is passed through unchanged
    out = er.redact_messages([{"role": "user", "content": 123}], mode="secrets")
    assert out[0]["content"] == 123


def test_mode_reads_config(monkeypatch):
    from app.core import config_loader as cl

    class _P:
        redact_egress = "strict"
    monkeypatch.setattr(cl.cfg, "privacy", _P(), raising=False)
    assert er.mode() == "strict"

    class _P2:
        redact_egress = "bogus"
    monkeypatch.setattr(cl.cfg, "privacy", _P2(), raising=False)
    assert er.mode() == "secrets"          # invalid → safe default
