"""Per-user tenancy scoping + Row-Level Security (vNext §10.2).

Two locks, defence in depth:

* **Application scoping** — every request resolves its ``user_id`` (JWT ``sub``
  when auth is on, else the device user) and the per-request DB transaction runs
  ``SET LOCAL app.user_id = '<uuid>'``.
* **Postgres RLS** — every user-owned table carries a policy
  ``USING (user_id = current_setting('app.user_id')::uuid)``, so the ONE query
  someone forgets to scope returns *nothing* instead of *everything*. When
  ``app.user_id`` is unset (or empty) the policy matches no rows — the fail-safe.

This module is the mechanism. The SQL builders are pure + injection-safe (the
value is re-parsed as a UUID, so arbitrary text can never reach the statement)
and unit-tested; applying RLS to the live schema is a guarded migration, and
wiring ``SET LOCAL`` into the per-request session is the integration step (both
need a running Postgres).
"""
from __future__ import annotations

import os
import uuid

# User-owned tables that already carry a ``user_id`` column and so can be put
# behind tenant RLS today. (Tables owned transitively — messages→sessions,
# agent_steps→agent_runs — are scoped via their parent's FK; the migration adds
# their own ``user_id`` where a direct policy is wanted.)
RLS_TABLES: tuple[str, ...] = (
    "resumes", "projects", "sessions", "model_usage",
    "episodes", "solve_sessions", "skills", "generated_documents",
)


def app_user_guc(user_id) -> str:
    """The ``SET LOCAL app.user_id = '<uuid>'`` statement RLS reads via
    ``current_setting('app.user_id')``.

    Injection-safe by construction: the value is parsed as a UUID and
    re-serialized, so only ``[0-9a-f-]`` can ever appear in the SQL. Raises
    ``ValueError`` on anything that isn't a UUID (the caller rejects the request
    rather than run an unscoped query)."""
    u = uuid.UUID(str(user_id))
    return f"SET LOCAL app.user_id = '{u}'"


def clear_app_user_guc() -> str:
    """Reset the scope (→ NULL → the RLS policy matches no rows)."""
    return "RESET app.user_id"


def rls_policy_statements(table: str) -> list[str]:
    """Idempotent DDL to put ``table`` behind STRICT tenant RLS. When
    ``app.user_id`` is unset/empty the ``NULLIF(...)::uuid`` is NULL, so
    ``user_id = NULL`` is false and NO rows are visible — a forgotten scope leaks
    nothing. Migrations / admin run as a ``BYPASSRLS`` role. ``FORCE`` subjects
    even the table owner to the policy."""
    if table not in RLS_TABLES:
        raise ValueError(f"{table!r} is not a known user-owned table")
    policy = f"{table}_tenant_isolation"
    return [
        f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY;',
        f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY;',
        f"DROP POLICY IF EXISTS {policy} ON \"{table}\";",
        f"CREATE POLICY {policy} ON \"{table}\" "
        f"USING (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);",
    ]


def all_rls_statements() -> list[str]:
    """Every table's RLS DDL — the body of the guarded migration."""
    out: list[str] = []
    for t in RLS_TABLES:
        out.extend(rls_policy_statements(t))
    return out


async def apply_user_scope(session, user_id) -> bool:
    """Run ``SET LOCAL app.user_id`` on ``session``'s current transaction. Call
    once per request after the transaction opens. Returns True when a scope was
    applied. Fail-open on a bad id (returns False; RLS then denies, which is the
    safe direction). ``session`` is an AsyncSession."""
    try:
        stmt = app_user_guc(user_id)
    except (ValueError, TypeError):
        return False
    try:
        from sqlalchemy import text
        await session.execute(text(stmt))
        return True
    except Exception:  # noqa: BLE001 — never let scoping crash a request
        return False


def rls_enabled() -> bool:
    """Whether tenant RLS is turned on. Same opt-in flag as the guarded
    migration (`0020`), so the policy DDL and the per-transaction scoping are
    enabled together. Default OFF → nothing changes."""
    return os.environ.get("TENANT_RLS_ENABLE") == "1"


def _resolved_scope_user() -> uuid.UUID | None:
    """The owner to scope THIS transaction to — the authenticated user when a
    request is signed in, else the (cached) device user. Sync so it can run in
    the `after_begin` event."""
    from storage.context import get_request_user_id
    uid = get_request_user_id()
    if uid is not None:
        return uid
    from storage.device import get_device_user_id
    return get_device_user_id()


def _scope_sql(uid: uuid.UUID | None) -> str:
    # A real uuid → the injection-safe SET LOCAL; no user → empty so the RLS
    # policy's NULLIF(...) is NULL and NO rows match (fail-safe deny).
    if uid is None:
        return "SET LOCAL app.user_id = ''"
    try:
        return app_user_guc(uid)
    except (ValueError, TypeError):
        return "SET LOCAL app.user_id = ''"


_owner_hook_installed = False


def install_owner_stamp_hook() -> None:
    """Stamp the owning ``user_id`` onto EVERY new user-owned row that doesn't
    already carry one (§10.1c) — so a chat/live session created anywhere (HTTP
    route, background task) belongs to the request's user, with no per-call-site
    change. Always on: the value is the authenticated user, or the device user
    when anonymous (identical to today for single-user)."""
    global _owner_hook_installed
    if _owner_hook_installed:
        return
    from sqlalchemy import event

    from storage.models import Session as _SessionRow

    @event.listens_for(_SessionRow, "before_insert")
    def _stamp_owner(mapper, connection, target):  # noqa: ANN001
        if getattr(target, "user_id", None) is None:
            uid = _resolved_scope_user()
            if uid is not None:
                target.user_id = uid

    _owner_hook_installed = True


_rls_hook_installed = False


def install_rls_session_hook() -> None:
    """Register a global ``after_begin`` listener that stamps
    ``SET LOCAL app.user_id`` onto EVERY transaction (so RLS policies filter to
    the request's user). Re-applying per-transaction is essential: ``SET LOCAL``
    is transaction-scoped, so a session that commits mid-request would otherwise
    lose its scope. Idempotent; a no-op at runtime unless ``rls_enabled()``."""
    global _rls_hook_installed
    if _rls_hook_installed:
        return
    from sqlalchemy import event
    from sqlalchemy.orm import Session as _SyncSession

    @event.listens_for(_SyncSession, "after_begin")
    def _set_tenant_scope(session, transaction, connection):  # noqa: ANN001
        if not rls_enabled():
            return
        try:
            connection.exec_driver_sql(_scope_sql(_resolved_scope_user()))
        except Exception:  # noqa: BLE001 — never let scoping crash a transaction
            pass

    _rls_hook_installed = True


async def ensure_supabase_user(user_id) -> uuid.UUID | None:
    """Idempotently upsert a ``users`` row for an authenticated Supabase user
    (keyed by the JWT ``sub`` UUID), so foreign keys resolve. Mirrors
    ``device.ensure_device_user`` for the account path. None on error."""
    try:
        uid = uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return None
    try:
        from sqlalchemy import select

        from storage.db import get_session_factory
        from storage.models import User
        factory = get_session_factory()
        if factory is None:
            return None
        async with factory() as session:
            row = await session.get(User, uid)
            if row is None:
                session.add(User(id=uid, preferences={"source": "supabase"}))
                await session.commit()
            return uid
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "RLS_TABLES", "app_user_guc", "clear_app_user_guc",
    "rls_policy_statements", "all_rls_statements", "apply_user_scope",
    "ensure_supabase_user", "rls_enabled", "install_rls_session_hook",
    "install_owner_stamp_hook",
]
