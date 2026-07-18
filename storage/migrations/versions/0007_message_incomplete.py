"""Add `incomplete` flag to messages.

Revision ID: 0007_message_incomplete
Revises: 0006_llm_vision
Create Date: 2026-06-01

Marks assistant turns that were cut short (client disconnect / Stop / a
provider dropping mid-stream). The partial text is still saved; the UI shows a
"Continue / Retry" affordance for these. Existing rows default to False.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0007_message_incomplete"
down_revision: Union[str, None] = "0006_llm_vision"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "incomplete",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "incomplete")
