"""Default device user — single-user-on-device support.

Per Architecture2.md §3 ("No auth, no sessions — device-local model"),
we don't have a login flow. But every row in the new schema can carry
a `user_id` so multi-user is wired up the day we add auth.

Today there's exactly one row in `users` per device, keyed by a
persistent install ID under `~/.zapthetrick/install_id`. The actual
upsert lives in [storage.device.ensure_device_user]; this module
preserves the legacy `ensure_default_user` / `get_default_user_id`
API so route handlers don't all have to change at once.
"""
from __future__ import annotations

import uuid

from .device import (
    ensure_device_user,
    get_device_user_id,
    reset_cache_for_tests as _reset_device_cache,
)


async def ensure_default_user() -> uuid.UUID | None:
    """Legacy alias — delegates to [ensure_device_user]."""
    return await ensure_device_user()


def get_default_user_id() -> uuid.UUID | None:
    """Legacy alias — delegates to [get_device_user_id]."""
    return get_device_user_id()


def reset_cache_for_tests() -> None:
    """Drop the in-process cache. Tests with multiple Postgres
    instances use this between fixtures."""
    _reset_device_cache()
