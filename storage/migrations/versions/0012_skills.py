"""skills + skill_bundles — the Antigravity skill library, loaded into Postgres
on startup (from the bundled skills_index.json + SKILL.md bodies + plugin
bundles). The Settings → Skills browser queries these tables (paginated, indexed)
instead of scanning the files each request.

One row per skill (metadata + the full SKILL.md body) and one per editorial
bundle (its member skill ids). Idempotent ingest lives in app/skills/store.py.

GUARDED CREATE: 0001_initial already creates a `skills` table for the MEMORY
system (session_tag/text/confidence — app/memory/semantic.py). Unconditionally
creating another `skills` here collided on every FRESH database, and because
the whole upgrade chain runs in one transaction, the failure rolled back ALL
tables (sessions included) silently on every boot. When `skills` already
exists we skip the library tables — the skills browser then uses its built-in
file fallback, which is the behavior every existing install has today.

Revision ID: 0012_skills
Revises: 0011_code_graphs
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_skills"
down_revision: Union[str, None] = "0011_code_graphs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())

    if not insp.has_table("skills"):
        op.create_table(
            "skills",
            # id = the skill's path (e.g. "skills/00-andruia-consultant") — unique.
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("skill_id", sa.Text(), nullable=False),  # short id (bundle refs)
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("category", sa.Text(), nullable=False, server_default="uncategorized"),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("risk", sa.Text(), nullable=True),
            sa.Column("source", sa.Text(), nullable=True),
            sa.Column("date_added", sa.Text(), nullable=True),
            sa.Column("targets", postgresql.JSONB(), nullable=True),
            sa.Column("body", sa.Text(), nullable=True),
            # Lowercased name+category+description for cheap ILIKE search.
            sa.Column("search", sa.Text(), nullable=False, server_default=""),
        )
        op.create_index("ix_skills_category", "skills", ["category"])
        op.create_index("ix_skills_skill_id", "skills", ["skill_id"])

    if not insp.has_table("skill_bundles"):
        op.create_table(
            "skill_bundles",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("skill_ids", postgresql.JSONB(), nullable=False),
            sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    # Best-effort: these may be the memory system's tables (0001) when the
    # guarded creates above were skipped — leave whatever exists alone only
    # if it predates this migration. Plain drops match the original revision.
    op.execute("DROP TABLE IF EXISTS skill_bundles")
    op.execute("DROP INDEX IF EXISTS ix_skills_skill_id")
    op.execute("DROP INDEX IF EXISTS ix_skills_category")
