"""Async repository over `llm_api_keys` — multi-key, encrypted at rest.

Keys are encrypted via `app.llm.crypto` before insert and returned masked
on read. The router calls `enabled_keys(platform)` to get the candidate
keys (enabled + healthy/unknown) for round-robin selection.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, update

from app.llm import crypto
from app.llm.catalog import get_provider_spec
from storage.db import get_session_factory
from storage.models import LLMApiKey


class KeysUnavailable(RuntimeError):
    """Raised when the DB isn't ready, so callers can surface a clear error."""


@dataclass
class KeyView:
    """Masked, UI-safe view of a stored key."""
    id: int
    platform: str
    label: str
    masked: str
    status: str
    enabled: bool


def _factory():
    factory = get_session_factory()
    if factory is None:
        raise KeysUnavailable("Database is not ready — cannot manage API keys yet.")
    return factory


async def add_key(platform: str, key: str, label: str = "") -> KeyView:
    """Encrypt and insert a new key for `platform`. Returns the masked view."""
    if get_provider_spec(platform) is None:
        raise ValueError(f"Unknown provider '{platform}'.")
    key = key.strip()
    if not key:
        raise ValueError("API key is empty.")
    # Self-heal a boot that skipped key init (Postgres up late): resolve the
    # persistent key now. Raises EncryptionKeyError only if the DB is STILL
    # unreachable — the route maps that to a clear 503.
    await crypto.ensure_initialized()
    enc, iv, tag = crypto.encrypt(key)
    async with _factory()() as session:
        row = LLMApiKey(
            platform=platform,
            label=label.strip(),
            encrypted_key=enc,
            iv=iv,
            auth_tag=tag,
            status="unknown",
            enabled=True,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return KeyView(row.id, row.platform, row.label, crypto.mask_key(key), row.status, row.enabled)


async def list_keys(platform: str | None = None) -> list[KeyView]:
    """All keys (optionally filtered by platform), masked for display."""
    try:
        await crypto.ensure_initialized()
    except Exception:  # noqa: BLE001 — degrade to "****" masks, don't fail the list
        pass
    stmt = select(LLMApiKey)
    if platform:
        stmt = stmt.where(LLMApiKey.platform == platform)
    stmt = stmt.order_by(LLMApiKey.platform, LLMApiKey.id)
    async with _factory()() as session:
        rows = (await session.execute(stmt)).scalars().all()
        out: list[KeyView] = []
        for r in rows:
            try:
                plain = crypto.decrypt(r.encrypted_key, r.iv, r.auth_tag)
                masked = crypto.mask_key(plain)
            except Exception:  # noqa: BLE001 — never let one bad row break the list
                masked = "****"
            out.append(KeyView(r.id, r.platform, r.label, masked, r.status, r.enabled))
        return out


async def set_enabled(key_id: int, enabled: bool) -> None:
    async with _factory()() as session:
        await session.execute(
            update(LLMApiKey).where(LLMApiKey.id == key_id).values(enabled=enabled)
        )
        await session.commit()


async def delete_key(key_id: int) -> None:
    async with _factory()() as session:
        row = await session.get(LLMApiKey, key_id)
        if row is not None:
            await session.delete(row)
            await session.commit()


async def key_counts() -> dict[str, dict]:
    """Per-platform {total, enabled, healthy} for the catalog screen."""
    async with _factory()() as session:
        rows = (await session.execute(select(LLMApiKey))).scalars().all()
    counts: dict[str, dict] = {}
    for r in rows:
        c = counts.setdefault(r.platform, {"total": 0, "enabled": 0, "healthy": 0})
        c["total"] += 1
        if r.enabled:
            c["enabled"] += 1
        if r.status == "healthy":
            c["healthy"] += 1
    return counts
