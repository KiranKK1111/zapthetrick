"""blobs — store uploaded + generated file bytes in Postgres.

Previously blobs (chat images, resumes, solve screenshots, exported documents)
lived on the filesystem and Postgres only held the path. This table holds the
bytes themselves so every upload AND every generated artifact persists in the
database and can be reloaded instantly — no external blob volume required for
the bundled, single-installer deployment.

Revision ID: 0010_blobs
Revises: 0009_pgvector
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010_blobs"
down_revision: Union[str, None] = "0009_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "blobs",
        # The caller-supplied relative key, e.g. "chat_images/<uuid>_name.png",
        # "documents/<uuid>.pdf". Primary key — put() upserts on it.
        sa.Column("path", sa.Text(), primary_key=True),
        sa.Column("filename", sa.Text(), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=True),
        # "upload" | "generated" (or finer: chat_image, resume, solve, document).
        sa.Column("kind", sa.Text(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("data", sa.LargeBinary(), nullable=False),  # bytea
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_blobs_kind", "blobs", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_blobs_kind", table_name="blobs")
    op.drop_table("blobs")
