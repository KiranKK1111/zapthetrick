"""Multi-provider LLM routing tables (freellmapi port).

Revision ID: 0005_llm_routing
Revises: 0004_continuity
Create Date: 2026-05-31

Six tables back `app/llm/*`: encrypted multi-key storage, the curated
model catalog, the fallback priority chain, the sliding-window rate-limit
ledger + cooldowns, and a small key/value store. Integer PKs (not UUID)
so the router's round-robin + fallback references stay cheap and mirror
the reference implementation.

The catalog rows + default fallback chain are NOT seeded here — they're
seeded idempotently from `app.llm.catalog.ensure_seeded()` on startup so
the curated list can evolve without a new migration.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0005_llm_routing"
down_revision: Union[str, None] = "0004_continuity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("platform", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=False, server_default=""),
        sa.Column("encrypted_key", sa.Text, nullable=False),
        sa.Column("iv", sa.Text, nullable=False),
        sa.Column("auth_tag", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="unknown"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("fail_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_llm_api_keys_platform", "llm_api_keys", ["platform"])

    op.create_table(
        "llm_models",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("platform", sa.Text, nullable=False),
        sa.Column("model_id", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("intelligence_rank", sa.Integer, nullable=False, server_default="100"),
        sa.Column("speed_rank", sa.Integer, nullable=False, server_default="100"),
        sa.Column("size_label", sa.Text, nullable=True),
        sa.Column("rpm_limit", sa.Integer, nullable=True),
        sa.Column("rpd_limit", sa.Integer, nullable=True),
        sa.Column("tpm_limit", sa.Integer, nullable=True),
        sa.Column("tpd_limit", sa.Integer, nullable=True),
        sa.Column("monthly_token_budget", sa.Text, nullable=True),
        sa.Column("context_window", sa.Integer, nullable=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
    )
    op.create_index(
        "ix_llm_models_platform_model", "llm_models", ["platform", "model_id"], unique=True
    )

    op.create_table(
        "llm_fallback_config",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "model_db_id",
            sa.Integer,
            sa.ForeignKey("llm_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("priority", sa.Integer, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
    )
    op.create_index("ix_llm_fallback_model", "llm_fallback_config", ["model_db_id"], unique=True)
    op.create_index("ix_llm_fallback_priority", "llm_fallback_config", ["priority"])

    op.create_table(
        "llm_rate_limit_usage",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("platform", sa.Text, nullable=False),
        sa.Column("model_id", sa.Text, nullable=False),
        sa.Column("key_id", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at_ms", sa.BigInteger, nullable=False),
    )
    op.create_index(
        "ix_llm_rl_usage_lookup",
        "llm_rate_limit_usage",
        ["platform", "model_id", "key_id", "kind", "created_at_ms"],
    )

    op.create_table(
        "llm_rate_limit_cooldowns",
        sa.Column("platform", sa.Text, primary_key=True),
        sa.Column("model_id", sa.Text, primary_key=True),
        sa.Column("key_id", sa.Integer, primary_key=True),
        sa.Column("expires_at_ms", sa.BigInteger, nullable=False),
    )
    op.create_index("ix_llm_rl_cooldowns_expires", "llm_rate_limit_cooldowns", ["expires_at_ms"])

    op.create_table(
        "llm_settings",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("llm_settings")
    op.drop_index("ix_llm_rl_cooldowns_expires", table_name="llm_rate_limit_cooldowns")
    op.drop_table("llm_rate_limit_cooldowns")
    op.drop_index("ix_llm_rl_usage_lookup", table_name="llm_rate_limit_usage")
    op.drop_table("llm_rate_limit_usage")
    op.drop_index("ix_llm_fallback_priority", table_name="llm_fallback_config")
    op.drop_index("ix_llm_fallback_model", table_name="llm_fallback_config")
    op.drop_table("llm_fallback_config")
    op.drop_index("ix_llm_models_platform_model", table_name="llm_models")
    op.drop_table("llm_models")
    op.drop_index("ix_llm_api_keys_platform", table_name="llm_api_keys")
    op.drop_table("llm_api_keys")
