"""episode project scoping — memory graph scoped to projects (Architecture §17).

Adds a nullable `project_id` FK to `episodes` so memory recall can scope across
every conversation in a project, not just one session. Purely additive: existing
episodes are ungrouped (project_id NULL) and recall by session_tag as today.
`ON DELETE SET NULL` so deleting a project detaches its episodes rather than
destroying them. Also indexes `episodes.user_id` for account-level export/delete
(§18).

Revision ID: 0016_episode_project_scope
Revises: 0015_projects
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016_episode_project_scope"
down_revision: Union[str, None] = "0015_projects"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "episodes",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_episodes_project_id", "episodes", "projects",
        ["project_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_episodes_project_id", "episodes", ["project_id"])
    op.create_index("ix_episodes_user_id", "episodes", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_episodes_user_id", table_name="episodes")
    op.drop_index("ix_episodes_project_id", table_name="episodes")
    op.drop_constraint("fk_episodes_project_id", "episodes", type_="foreignkey")
    op.drop_column("episodes", "project_id")
