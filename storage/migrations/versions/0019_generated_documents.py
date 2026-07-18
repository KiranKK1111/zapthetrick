"""versioned generated-document artifacts (Document Generation roadmap, Phase 5).

Adds `generated_documents`: a versioned store for produced documents. Each row
holds the SOURCE markdown (every export format + the structured model derive from
it) plus metadata; an edit creates a new row sharing `doc_key` with `version`+1,
giving an evolution timeline + incremental updates without a full regeneration.

Idempotent (CREATE TABLE / INDEX IF NOT EXISTS) so a partially-migrated or
already-current DB re-runs safely on startup.

Revision ID: 0019_generated_documents
Revises: 0018_rename_vector_point_id
"""
from __future__ import annotations

from typing import Union

from alembic import op

revision: str = "0019_generated_documents"
down_revision: Union[str, None] = "0018_rename_vector_point_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_documents (
            id           UUID PRIMARY KEY,
            session_id   UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            doc_key      UUID NOT NULL,
            version      INTEGER NOT NULL DEFAULT 1,
            title        TEXT NOT NULL DEFAULT '',
            doc_format   VARCHAR(16) NOT NULL DEFAULT 'pdf',
            goal         VARCHAR(40),
            content_md   TEXT NOT NULL DEFAULT '',
            meta         JSONB,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_generated_documents_session_id "
        "ON generated_documents (session_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_generated_documents_doc_key "
        "ON generated_documents (doc_key);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_generated_documents_key_version "
        "ON generated_documents (doc_key, version);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS generated_documents;")
