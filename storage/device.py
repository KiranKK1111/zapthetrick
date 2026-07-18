"""Device install ID — stable identity for the local-only mode.

Architecture2.md §3 ships the app as "no auth, no sessions — device-
local model". The legacy [get_default_user_id] looked up a JSONB
marker on every fresh process and re-created the row when the cache
was empty. That works but means two cold starts on the same machine
*can* create two `users` rows if the marker is wiped.

This module gives us a stronger guarantee: a UUID is generated once
at first run, written to disk under `~/.zapthetrick/install_id`,
and reused thereafter. The DB row is upserted by that ID — never
duplicated.

Resolution order:
    1. `ZAPTHETRICK_DEVICE_ID` env var (tests / CI override)
    2. The file `~/.zapthetrick/install_id`
    3. Generate new UUID4, write the file, return it

The DB-side helper [ensure_device_user] looks up (or inserts) a row
in `users` whose `preferences.device_install_id` matches.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import select

from .db import get_session_factory
from .models import User


log = logging.getLogger(__name__)


_INSTALL_ID_ENV = "ZAPTHETRICK_DEVICE_ID"
_INSTALL_ID_FILE = Path.home() / ".zapthetrick" / "install_id"


def device_install_id() -> str:
    """Return the persistent device install ID. Creates one on first
    run. Never raises — falls back to an in-memory UUID if the home
    directory isn't writable (sandbox, container without home, …).
    """
    env = os.environ.get(_INSTALL_ID_ENV)
    if env:
        return env.strip()

    try:
        if _INSTALL_ID_FILE.exists():
            existing = _INSTALL_ID_FILE.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except OSError as exc:
        log.warning("could not read install_id file: %s", exc)

    new_id = str(uuid.uuid4())
    try:
        _INSTALL_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _INSTALL_ID_FILE.write_text(new_id, encoding="utf-8")
        log.info("created new device install_id at %s", _INSTALL_ID_FILE)
    except OSError as exc:
        log.warning(
            "could not persist install_id (using in-memory): %s", exc
        )
    return new_id


_cached_user_id: uuid.UUID | None = None


async def ensure_device_user() -> uuid.UUID | None:
    """Find or upsert the `users` row keyed by the device install ID.

    Idempotent. Two cold starts on the same machine return the same
    UUID. Failures are non-fatal — falls back to None, route handlers
    treat that as anonymous mode (the `user_id` column is nullable).
    """
    global _cached_user_id
    if _cached_user_id is not None:
        return _cached_user_id

    factory = get_session_factory()
    if factory is None:
        return None

    install_id = device_install_id()
    try:
        async with factory() as session:
            # Marker query: row whose preferences carry the install_id.
            row = (
                await session.execute(
                    select(User).where(
                        User.preferences["device_install_id"].astext == install_id
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                # Migrate the legacy `{"device": true}` row if one
                # exists — single-user-on-device should remain a
                # single row across the upgrade.
                row = (
                    await session.execute(
                        select(User).where(
                            User.preferences["device"].astext == "true"
                        )
                    )
                ).scalar_one_or_none()
                if row is not None:
                    prefs = dict(row.preferences or {})
                    prefs["device_install_id"] = install_id
                    row.preferences = prefs
                else:
                    row = User(
                        preferences={
                            "device": True,
                            "device_install_id": install_id,
                        }
                    )
                    session.add(row)
                await session.commit()
                await session.refresh(row)
            _cached_user_id = row.id
            return _cached_user_id
    except Exception as exc:  # noqa: BLE001
        log.warning("ensure_device_user failed (anonymous mode): %s", exc)
        return None


def get_device_user_id() -> uuid.UUID | None:
    """Cached lookup — call this from routes / agents."""
    return _cached_user_id


def reset_cache_for_tests() -> None:
    global _cached_user_id
    _cached_user_id = None


__all__ = [
    "device_install_id",
    "ensure_device_user",
    "get_device_user_id",
    "reset_cache_for_tests",
]
