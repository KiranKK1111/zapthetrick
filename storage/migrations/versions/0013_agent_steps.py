"""agent_steps — the ordered, replayable trace of a Code-In agent run.

One row per SSE event the agent loop emits (thought / tool_call / tool_result /
approval / final / error / goal_* / skill), hanging off a `Session`
(`type="agent_code"`). Lets a past agent session reload its full step-by-step
trace. Kept separate from `messages` so listing sessions stays a no-join query
and these high-cardinality rows never bloat the message FTS index.

Revision ID: 0013_agent_steps
Revises: 0012_skills
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_agent_steps"
down_revision: Union[str, None] = "0012_skills"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("turn", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event", sa.String(24), nullable=False),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("tool", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(16), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("incomplete", sa.Boolean(), nullable=False,
                  server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_steps_session_seq", "agent_steps",
                    ["session_id", "seq"])


def downgrade() -> None:
    op.drop_index("ix_agent_steps_session_seq", table_name="agent_steps")
    op.drop_table("agent_steps")
