"""Choose a [VectorStore] backend from config.

Everything lives in Postgres via pgvector by default (embeddings + RAG vectors
+ BM25 in one store). `chroma` is kept as an alternative embedded path. There is
no separate vector database.

Selection rule:
  - `vector_store.provider == 'chroma'` → [ChromaStore].
  - Otherwise → [PgVectorStore] (the default).
"""
from __future__ import annotations

from app.core.config_loader import cfg

from .base import VectorStore
from .chroma_store import ChromaStore


_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the configured store. Cached at module level for the
    lifetime of the process (the underlying client manages its own
    pooling)."""
    global _singleton
    if _singleton is not None:
        return _singleton

    provider = (cfg.vector_store.provider or "").lower()
    if provider == "chroma":
        _singleton = ChromaStore(persist_dir=cfg.vector_store.persist_dir)
    else:
        # Default: everything in Postgres via pgvector — no separate vector DB.
        from .pgvector_store import PgVectorStore

        _singleton = PgVectorStore()
    return _singleton


async def close_vector_store() -> None:
    global _singleton
    if _singleton is not None:
        close = getattr(_singleton, "close", None)
        if close is not None:
            await close()
    _singleton = None


def reset() -> None:
    """Drop the cached singleton without awaiting close().

    Used by the live-config event bus when the user re-points the vector
    store. The next `get_vector_store()` builds a fresh store against the
    new cfg.
    """
    global _singleton
    _singleton = None
