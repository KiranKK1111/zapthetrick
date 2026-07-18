"""AES-256-GCM at-rest encryption for provider API keys.

Ported from freellmapi's `lib/crypto.ts`. Keys are encrypted before they
touch the `llm_api_keys` table and decrypted only in-memory at request
time (in the router). We store the ciphertext, the nonce (`iv`), and the
GCM auth tag in separate columns — same shape as the reference.

Key source, in order:
  1. env `ZAPTHETRICK_ENCRYPTION_KEY` — 64 hex chars (32 bytes). Required
     in production.
  2. dev fallback — when `ZAPTHETRICK_DEV_MODE=true`, a key is generated
     once and persisted to the `llm_settings` table (key='encryption_key').

This is symmetric and self-contained: we both encrypt and decrypt in
Python, so we don't need byte-for-byte parity with the Node implementation.
"""
from __future__ import annotations

import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


KEY_BYTES = 32
KEY_HEX_LEN = KEY_BYTES * 2
NONCE_BYTES = 12  # standard GCM nonce
_ENV_VAR = "ZAPTHETRICK_ENCRYPTION_KEY"
_PLACEHOLDER = "your-64-char-hex-key-here"

_cached_key: bytes | None = None


class EncryptionKeyError(RuntimeError):
    """Raised when no usable encryption key can be resolved."""


def _parse_hex_key(value: str, source: str) -> bytes:
    value = value.strip()
    if len(value) != KEY_HEX_LEN or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise EncryptionKeyError(
            f"Invalid {_ENV_VAR} ({source}): expected {KEY_HEX_LEN} hex chars "
            f"(32 bytes), got {len(value)}. Generate one with: "
            f'python -c "import secrets; print(secrets.token_hex(32))"'
        )
    return bytes.fromhex(value)


def _dev_mode() -> bool:
    return os.environ.get("ZAPTHETRICK_DEV_MODE", "").lower() == "true"


async def init_encryption_key() -> None:
    """Resolve and cache the encryption key. Call after the DB is up.

    Env var wins. Otherwise, in dev mode, a key is generated and persisted
    to `llm_settings`. Outside dev mode with no env var this raises so a
    misconfigured prod fails fast instead of silently dropping keys.
    """
    global _cached_key

    env_key = os.environ.get(_ENV_VAR)
    if env_key and env_key != _PLACEHOLDER:
        _cached_key = _parse_hex_key(env_key, "env")
        return

    # No env key: auto-generate one and persist it to the DB. ZapTheTrick is a
    # local single-user app, so "just works" beats "fails until you export a
    # 64-char hex var". Production should still set ZAPTHETRICK_ENCRYPTION_KEY
    # (logged below) so keys aren't tied to the DB row.
    import logging

    logging.getLogger(__name__).info(
        "%s not set — using an auto-generated key persisted in the DB. "
        "Set %s (64 hex chars) for a portable production key.",
        _ENV_VAR, _ENV_VAR,
    )

    from sqlalchemy import select

    from storage.db import get_session_factory
    from storage.models import LLMSetting

    factory = get_session_factory()
    if factory is None:
        # DB not ready — do NOT fabricate an ephemeral key: anything encrypted
        # under it would be silently orphaned on restart, and existing keys
        # would fail to decrypt for the whole session. Leave the key
        # uninitialized; ensure_initialized() retries on first use.
        raise EncryptionKeyError(
            "Database not ready — cannot resolve the persistent encryption "
            "key yet. Retried automatically on first use."
        )

    async with factory() as session:
        row = (
            await session.execute(
                select(LLMSetting).where(LLMSetting.key == "encryption_key")
            )
        ).scalar_one_or_none()
        if row is not None:
            _cached_key = _parse_hex_key(row.value, "db")
            return
        key = secrets.token_bytes(KEY_BYTES)
        session.add(LLMSetting(key="encryption_key", value=key.hex()))
        await session.commit()
        _cached_key = key


async def ensure_initialized() -> None:
    """Lazy init for callers that may run before startup finished — or after
    a boot where Postgres came up late and the startup init was skipped.
    No-op once a key is cached; raises EncryptionKeyError while the DB is
    still unreachable (callers retry on their next use)."""
    if _cached_key is None:
        await init_encryption_key()


def _get_key() -> bytes:
    if _cached_key is None:
        raise EncryptionKeyError(
            "Encryption key not initialized. Call init_encryption_key() first."
        )
    return _cached_key


def encrypt(plaintext: str) -> tuple[str, str, str]:
    """Return (ciphertext_hex, iv_hex, auth_tag_hex)."""
    key = _get_key()
    nonce = secrets.token_bytes(NONCE_BYTES)
    sealed = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    # AESGCM appends the 16-byte tag to the ciphertext; split it out so the
    # three columns mirror the freellmapi schema.
    ciphertext, tag = sealed[:-16], sealed[-16:]
    return ciphertext.hex(), nonce.hex(), tag.hex()


def decrypt(ciphertext_hex: str, iv_hex: str, auth_tag_hex: str) -> str:
    key = _get_key()
    sealed = bytes.fromhex(ciphertext_hex) + bytes.fromhex(auth_tag_hex)
    plaintext = AESGCM(key).decrypt(bytes.fromhex(iv_hex), sealed, None)
    return plaintext.decode("utf-8")


def mask_key(key: str) -> str:
    """`sk-or-v1-…abcd` style mask for safe display."""
    if len(key) <= 8:
        return "****" + key[-4:]
    return key[:4] + "…" + key[-4:]
