"""Add rolling-summary columns to sessions.

Revision ID: 0008_session_summary
Revises: 0007_message_incomplete
Create Date: 2026-06-01

A very long conversation can't send its entire history to the model every turn
(latency + cost grow linearly, and it eventually blows the context window). We
keep the recent turns verbatim (token-budgeted window) plus a rolling SUMMARY
of the older turns. `summary` holds that condensed text; `summary_count` is how
many of the oldest messages it already covers. Existing rows default to empty.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0008_session_summary"
down_revision: Union[str, None] = "0007_message_incomplete"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "summary_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "summary_count")
    op.drop_column("sessions", "summary")
