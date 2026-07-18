"""Continuity tables — session_links + session_topics.

Revision ID: 0004_continuity
Revises: 0003_chat_history
Create Date: 2026-05-15

Architecture.md §"Conversation link graph" introduces two tables the
continuity layer relies on. They used to be created lazily on first
write via `ensure_schema()`; that's fragile for read-only roles and
makes the schema implicit. This migration owns them properly.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0004_continuity"
down_revision: Union[str, None] = "0003_chat_history"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # session_links — explicit edges between sessions.
    op.create_table(
        "session_links",
        sa.Column("from_session", UUID(as_uuid=True), nullable=False),
        sa.Column("to_session", UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.Text, nullable=False, server_default="references"),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=False, server_default="0.5"),
        sa.Column("rationale", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("from_session", "to_session", "kind"),
    )
    op.create_index("ix_session_links_from", "session_links", ["from_session"])
    op.create_index("ix_session_links_to", "session_links", ["to_session"])

    # session_topics — topic label per session.
    op.create_table(
        "session_topics",
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column(
            "keywords",
            sa.dialects.postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_session_topics_topic", "session_topics", ["topic"])


def downgrade() -> None:
    op.drop_index("ix_session_topics_topic", table_name="session_topics")
    op.drop_table("session_topics")
    op.drop_index("ix_session_links_to", table_name="session_links")
    op.drop_index("ix_session_links_from", table_name="session_links")
    op.drop_table("session_links")
