"""message envelope — persist the unified response.v1 envelope per turn.

Adds a nullable JSONB `envelope` column to `messages` so a reload reconstructs
the same canonical object the turn streamed live (Architecture.md §5). Purely
additive: existing rows are NULL and the load path reconstructs a minimal
envelope from the other columns.

Revision ID: 0014_message_envelope
Revises: 0013_agent_steps
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_message_envelope"
down_revision: Union[str, None] = "0013_agent_steps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("envelope", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "envelope")
