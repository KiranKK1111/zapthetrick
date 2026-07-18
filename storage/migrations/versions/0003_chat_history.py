"""Add chat-history curation columns to sessions.

Revision ID: 0003_chat_history
Revises: 0002_solve_sessions
Create Date: 2026-05-15

Architecture2.md §"Chat tab — history" calls for pin / archive / tag /
search affordances on the conversations list. This migration adds the
backing columns and the composite indexes the list query relies on.

`message_count` and `last_message_at` are cached on the session row so
the drawer can render hundreds of sessions without a per-row join. A
backfill statement walks the existing `messages` table once.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0003_chat_history"
down_revision: Union[str, None] = "0002_solve_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "sessions",
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "sessions",
        sa.Column("tags", sa.dialects.postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "message_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "last_message_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Backfill from messages — one pass is plenty; the drawer keys on
    # these columns from now on.
    op.execute(
        """
        UPDATE sessions s
        SET message_count = sub.cnt,
            last_message_at = sub.last_at
        FROM (
            SELECT session_id,
                   COUNT(*)        AS cnt,
                   MAX(created_at) AS last_at
            FROM messages
            GROUP BY session_id
        ) sub
        WHERE s.id = sub.session_id
        """
    )

    op.create_index(
        "ix_sessions_user_pinned_updated",
        "sessions",
        ["user_id", "pinned", "updated_at"],
    )
    op.create_index(
        "ix_sessions_user_archived_updated",
        "sessions",
        ["user_id", "archived", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_user_archived_updated", table_name="sessions")
    op.drop_index("ix_sessions_user_pinned_updated", table_name="sessions")
    op.drop_column("sessions", "last_message_at")
    op.drop_column("sessions", "message_count")
    op.drop_column("sessions", "tags")
    op.drop_column("sessions", "archived")
    op.drop_column("sessions", "pinned")
