"""Add supports_vision flag to llm_models.

Revision ID: 0006_llm_vision
Revises: 0005_llm_routing
Create Date: 2026-06-01

Lets the router send image-bearing chat turns only to vision-capable models
(gemini-2.5-*, gpt-5, llama-4 variants, …) while keeping the same rank-based
fallback. Existing rows default to False; `catalog.seed_provider()` sets the
flag for the curated vision models on the next key add.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0006_llm_vision"
down_revision: Union[str, None] = "0005_llm_routing"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_models",
        sa.Column(
            "supports_vision",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("llm_models", "supports_vision")
