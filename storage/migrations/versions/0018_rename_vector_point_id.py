"""rename qdrant_point_id -> vector_point_id (drop Qdrant naming from the DB).

The vector store is Postgres/pgvector; the per-row point identifier column was
historically named `qdrant_point_id`. Qdrant is no longer part of the stack, so
this renames the column to the neutral `vector_point_id` on every table that
carries it (resume_chunks, episodes, skills). Pure rename — data preserved,
dependent indexes follow the column automatically in Postgres.

Revision ID: 0018_rename_vector_point_id
Revises: 0017_skill_project_scope
"""
from __future__ import annotations

from typing import Union

from alembic import op

revision: str = "0018_rename_vector_point_id"
down_revision: Union[str, None] = "0017_skill_project_scope"
branch_labels = None
depends_on = None

_TABLES = ("resume_chunks", "episodes", "skills")


def _rename(table: str, old: str, new: str) -> None:
    # Idempotent + safe: only rename when `old` exists and `new` doesn't, so a
    # partially-migrated or already-neutral DB doesn't error.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name = '{old}'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name = '{new}'
            ) THEN
                ALTER TABLE {table} RENAME COLUMN {old} TO {new};
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    for t in _TABLES:
        _rename(t, "qdrant_point_id", "vector_point_id")


def downgrade() -> None:
    for t in _TABLES:
        _rename(t, "vector_point_id", "qdrant_point_id")
