"""Consolidate RAG vectors into Postgres via pgvector.

Revision ID: 0009_pgvector
Revises: 0008_session_summary
Create Date: 2026-06-01

One store for embeddings + RAG vectors + keyword search, instead of Qdrant /
Chroma alongside Postgres. `rag_vectors` holds every chunk's embedding AND its
text, so dense (pgvector HNSW, cosine) and sparse (BM25 via a generated
`tsvector` + GIN) both run in one table — that's the hybrid retrieval path.

Dimension is 1024 to match the upgraded embedding model (BAAI/bge-m3). Changing
to a model with a different dimension needs a new migration (the HNSW index is
typed on the dimension).

Requires the `vector` extension to be available in the Postgres image. The
CREATE EXTENSION below installs it (needs a superuser/owner role — the app's
`postgres` user qualifies).
"""
from __future__ import annotations

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0009_pgvector"
down_revision: Union[str, None] = "0008_session_summary"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DIM = 1024  # BAAI/bge-m3 dense dimension
log = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    # pgvector must be INSTALLED in the Postgres image for `CREATE EXTENSION
    # vector` to work. If it isn't available, skip this migration gracefully
    # instead of aborting the whole startup — the app then boots without
    # pgvector RAG (retrieval degrades to empty), and the store will create the
    # table on first use once pgvector is installed. See app/.../pgvector_store.
    bind = op.get_bind()
    available = bind.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    ).scalar()
    if not available:
        log.warning(
            "pgvector ('vector') extension is NOT installed in this Postgres — "
            "skipping rag_vectors creation. Install pgvector (>=0.5) and restart "
            "to enable RAG; until then document retrieval returns no hits."
        )
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rag_vectors (
            id          UUID PRIMARY KEY,
            collection  TEXT NOT NULL,
            content     TEXT NOT NULL,
            payload     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding   vector({_DIM}) NOT NULL,
            content_tsv tsvector GENERATED ALWAYS AS
                            (to_tsvector('english', content)) STORED,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_vectors_collection "
        "ON rag_vectors (collection)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_vectors_tsv "
        "ON rag_vectors USING gin (content_tsv)"
    )
    # HNSW for cosine ANN. m / ef_construction left at sensible defaults.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_vectors_embedding "
        "ON rag_vectors USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rag_vectors")
    # Leave the `vector` extension installed — other tables may use it.
