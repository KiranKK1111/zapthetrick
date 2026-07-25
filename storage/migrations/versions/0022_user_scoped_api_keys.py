"""Per-user provider API keys (vNext §10.1c/§10.2).

Adds `user_id` to `llm_api_keys` so each account brings its own provider keys
instead of the whole deployment sharing one set. NULL = a legacy/global key,
which is what anonymous (auth-off) mode keeps using — so existing single-user
deployments are unaffected until accounts are turned on.

Additive + reversible: the column is nullable, existing rows stay NULL.

Revision ID: 0022_user_scoped_api_keys
Revises: 0021_user_auth_columns
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0022_user_scoped_api_keys"
down_revision: Union[str, None] = "0021_user_auth_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_api_keys",
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_llm_api_keys_user_platform", "llm_api_keys", ["user_id", "platform"]
    )


def downgrade() -> None:
    op.drop_index("ix_llm_api_keys_user_platform", table_name="llm_api_keys")
    op.drop_column("llm_api_keys", "user_id")
