"""code_graphs — persisted code knowledge graphs built from uploaded project
archives (zip/tar/…). One row per built graph, scoped to a conversation.

The full node/edge set is stored as JSONB (`graph`) so a follow-up turn can
reload it and run query tools (callers/callees/impact/…), plus a precomputed
`summary` (the project overview injected into the answer context) and headline
counts for quick listing. Modelled on codegraph's nodes/edges, kept in one row
for simplicity — can be normalised into node/edge tables later for SQL-level
graph queries.

Revision ID: 0011_code_graphs
Revises: 0010_blobs
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_code_graphs"
down_revision: Union[str, None] = "0010_blobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "code_graphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        # Stringified conversation/session id (no hard FK — cleaned up by the
        # delete-conversation handler, same as blobs).
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("files_parsed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("nodes_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edges_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("languages", postgresql.JSONB(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        # {"nodes": [...], "edges": [...]} — the full graph for reload + queries.
        sa.Column("graph", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_code_graphs_conversation", "code_graphs",
                    ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_code_graphs_conversation", table_name="code_graphs")
    op.drop_table("code_graphs")
