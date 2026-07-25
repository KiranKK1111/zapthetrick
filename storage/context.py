"""Per-request user context (vNext §10.1c/§10.2).

A neutral, low-level home for the authenticated user id so BOTH the API layer
(which SETS it, in `app.api.auth.AuthMiddleware`) and the storage layer (which
READS it, in `ensure_device_user` / `get_default_user_id`) can share it without
a storage→app import. This is what makes every route scope to the logged-in
user with no per-call-site changes: the existing device-user resolvers simply
return the auth user when one is present, and fall back to the device user
otherwise (auth-off → byte-identical to today).
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar

# The verified user id (a str `sub`/UUID) for the current request, or None when
# anonymous / auth-off. Set once at the edge by the auth middleware.
current_user_id_var: ContextVar[str | None] = ContextVar(
    "zaptrick_user_id", default=None)


def get_request_user_id() -> uuid.UUID | None:
    """The current request's authenticated user as a UUID, or None."""
    raw = current_user_id_var.get()
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError):
        return None
