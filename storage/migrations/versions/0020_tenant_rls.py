"""Per-user Row-Level Security on the user-owned tables (vNext §10.2).

Puts every user-owned table (`storage.tenancy.RLS_TABLES`) behind a strict tenant
policy: ``USING (user_id = current_setting('app.user_id')::uuid)`` — so the one
query someone forgets to scope returns *nothing* instead of *everything*.

SAFETY — this migration is a **no-op unless ``TENANT_RLS_ENABLE=1``**. Enabling
RLS requires the per-request ``SET LOCAL app.user_id`` scoping
(``storage.tenancy.apply_user_scope`` wired into the session) to be active — else
every query would match zero rows and the app would appear empty. So it stays
inert until you opt in, at which point it applies the tested policy DDL. Fully
reversible.

Revision ID: 0020_tenant_rls
Revises: 0019_generated_documents
"""
from __future__ import annotations

import os
from typing import Union

from alembic import op

revision: str = "0020_tenant_rls"
down_revision: Union[str, None] = "0019_generated_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("TENANT_RLS_ENABLE") != "1":
        # Opt-in only — see the module docstring. Safe no-op by default so this
        # migration can ship without changing any current (device-mode) deploy.
        return
    from storage.tenancy import all_rls_statements
    for stmt in all_rls_statements():
        op.execute(stmt)


def downgrade() -> None:
    from storage.tenancy import RLS_TABLES
    for t in RLS_TABLES:
        op.execute(f'ALTER TABLE "{t}" DISABLE ROW LEVEL SECURITY;')
        op.execute(f'DROP POLICY IF EXISTS {t}_tenant_isolation ON "{t}";')
