"""Native email/password account columns on `users` (vNext §10.1c).

Adds `email` (unique), `password_hash`, `name`, `email_verified` so the backend
can host its own accounts (self-minted HS256 JWTs) — no external identity
provider required, works identically in local and RunPod deploys.

SAFE + additive: every column is nullable (or has a server default), so existing
device-user rows keep working untouched and auth stays OFF until a secret is
configured. Fully reversible.

Revision ID: 0021_user_auth_columns
Revises: 0020_tenant_rls
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0021_user_auth_columns"
down_revision: Union[str, None] = "0020_tenant_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("email", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("password_hash", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("name", sa.Text(), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # Case-insensitive uniqueness on non-null emails (device rows have NULL email
    # and are exempt). A partial unique index keeps multiple NULLs legal.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email_lower "
        "ON users (lower(email)) WHERE email IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_users_email_lower")
    op.drop_column("users", "email_verified")
    op.drop_column("users", "name")
    op.drop_column("users", "password_hash")
    op.drop_column("users", "email")
