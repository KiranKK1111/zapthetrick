"""Per-user model catalog + fallback config (vNext §10.1c).

Each account gets its OWN `llm_models` (catalog + enable/disable) and
`llm_fallback_config` (priority), seeded on key-add. The shared on-pod `local`
floor stays NULL-owned (one physical model for everyone).

Crucially DROPS the old GLOBAL unique `(platform, model_id)` on `llm_models` —
with per-user catalogs the same model exists once per user, so the global unique
would reject every user after the first. Replaced by:
  * a partial unique on `(user_id, platform, model_id)` for owned rows, and
  * a partial unique on `(platform, model_id)` for the shared NULL rows.

Additive + reversible.

Revision ID: 0023_user_scoped_models
Revises: 0022_user_scoped_api_keys
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0023_user_scoped_models"
down_revision: Union[str, None] = "0022_user_scoped_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("llm_models", "llm_fallback_config"):
        op.add_column(
            table,
            sa.Column(
                "user_id", UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        )

    # Drop the GLOBAL unique so the same (platform, model_id) can exist per user.
    op.execute("DROP INDEX IF EXISTS ix_llm_models_platform_model")
    op.create_index(
        "ix_llm_models_user_platform", "llm_models", ["user_id", "platform"])
    # Per-user uniqueness for owned rows...
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_llm_models_user_platform_model "
        "ON llm_models (user_id, platform, model_id) WHERE user_id IS NOT NULL")
    # ...and keep the shared (NULL-owned, e.g. local floor) rows unique too.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_llm_models_shared_platform_model "
        "ON llm_models (platform, model_id) WHERE user_id IS NULL")

    op.create_index(
        "ix_llm_fallback_user", "llm_fallback_config", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_fallback_user", table_name="llm_fallback_config")
    op.execute("DROP INDEX IF EXISTS ux_llm_models_shared_platform_model")
    op.execute("DROP INDEX IF EXISTS ux_llm_models_user_platform_model")
    op.drop_index("ix_llm_models_user_platform", table_name="llm_models")
    op.create_index(
        "ix_llm_models_platform_model", "llm_models",
        ["platform", "model_id"], unique=True)
    op.drop_column("llm_fallback_config", "user_id")
    op.drop_column("llm_models", "user_id")
