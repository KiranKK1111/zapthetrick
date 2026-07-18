"""skill project scoping — semantic memory scoped to projects (Architecture §17).

Mirrors 0016 for the `skills` table: a nullable `project_id` FK so skill recall
can scope across every conversation in a project. Additive; `ON DELETE SET NULL`
so deleting a project detaches its skills rather than destroying them. Also
indexes `skills.user_id` for account-level export/delete (§18).

Revision ID: 0017_skill_project_scope
Revises: 0016_episode_project_scope
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_skill_project_scope"
down_revision: Union[str, None] = "0016_episode_project_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_skills_project_id", "skills", "projects",
        ["project_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_skills_project_id", "skills", ["project_id"])
    op.create_index("ix_skills_user_id", "skills", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_skills_user_id", table_name="skills")
    op.drop_index("ix_skills_project_id", table_name="skills")
    op.drop_constraint("fk_skills_project_id", "skills", type_="foreignkey")
    op.drop_column("skills", "project_id")
