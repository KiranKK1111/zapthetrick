"""projects — group conversations + scope their graphs (Architecture §17).

Creates the `projects` table and adds a nullable `project_id` FK to `sessions`.
Purely additive: existing sessions are ungrouped (project_id NULL) and behave
exactly as today. Deleting a project sets its sessions' project_id back to NULL
(ON DELETE SET NULL) — conversations are never destroyed with the project.

Revision ID: 0015_projects
Revises: 0014_message_envelope
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_projects"
down_revision: Union[str, None] = "0014_message_envelope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.Text(), nullable=False,
                  server_default="New project"),
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("archived", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_projects_user_updated", "projects",
                    ["user_id", "updated_at"])

    op.add_column(
        "sessions",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_sessions_project_id", "sessions", "projects",
        ["project_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_sessions_project_id", "sessions", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_sessions_project_id", table_name="sessions")
    op.drop_constraint("fk_sessions_project_id", "sessions", type_="foreignkey")
    op.drop_column("sessions", "project_id")
    op.drop_index("ix_projects_user_updated", table_name="projects")
    op.drop_table("projects")
