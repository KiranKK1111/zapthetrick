"""Add solve_sessions table for Solve history.

Revision ID: 0002_solve_sessions
Revises: 0001_initial
Create Date: 2026-05-15

Captures every Solve-screen click as a row so the history drawer can
reload a past problem + response, just like the Chat tab's
conversations list.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0002_solve_sessions"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "solve_sessions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(200), nullable=False, server_default="Untitled solve"),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("response", sa.Text, nullable=False, server_default=""),
        sa.Column("language", sa.String(50), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="text"),
        sa.Column("image_path", sa.Text, nullable=True),
        sa.Column("vision_model", sa.Text, nullable=True),
        sa.Column("code_model", sa.Text, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_solve_sessions_user_id", "solve_sessions", ["user_id"])
    op.create_index("ix_solve_sessions_created_at", "solve_sessions", ["created_at"])


def downgrade() -> None:
    op.drop_table("solve_sessions")
