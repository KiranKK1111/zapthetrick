"""Postgres + pgvector vector store.

One store for embeddings + RAG vectors + keyword search. Every chunk lives in
the `rag_vectors` table (created by migration 0009) with BOTH its embedding
(`vector(1024)`, HNSW cosine) and its text (`content` + a generated `tsvector`,
GIN). So this backend serves:

  * dense ANN      — `query()`        (embedding `<=>` cosine distance)
  * sparse BM25    — `_bm25()`        (`websearch_to_tsquery` + `ts_rank`)
  * hybrid         — `hybrid_query()` (dense + sparse fused with RRF)

Implements the [VectorStore] protocol, so it drops in via the factory with no
changes to the RAG callers. Uses the app's existing async engine pool.
"""
from __future__ import annotations

import json
import uuid

from sqlalchemy import text

from .base import Hit

_DIM = 1024  # must match migration 0009 + the configured embedding model


def _vec_literal(vector: list[float]) -> str:
    """pgvector text input form: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7f}" for x in vector) + "]"


def _as_payload(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


class PgVectorStore:
    """pgvector-backed store. Collections are just a `collection` column value
    in the shared `rag_vectors` table — no per-collection DDL."""

    def __init__(self) -> None:
        self._ensured: set[str] = set()
        self._table_ready = False

    def _factory(self):
        from storage.db import get_session_factory

        f = get_session_factory()
        if f is None:
            raise RuntimeError("Database not ready — no session factory.")
        return f

    async def _ensure_table(self) -> None:
        """Idempotently create the extension + table + indexes on first use.

        Mirrors migration 0009, but also covers the case where pgvector was
        installed AFTER that migration ran (the migration no-ops when the
        extension is missing). Raises if pgvector still isn't available —
        callers treat that as "store unavailable" and degrade to no hits.
        """
        if self._table_ready:
            return
        async with self._factory()() as s:
            await s.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await s.execute(
                text(
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
            )
            await s.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rag_vectors_collection "
                    "ON rag_vectors (collection)"
                )
            )
            await s.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rag_vectors_tsv "
                    "ON rag_vectors USING gin (content_tsv)"
                )
            )
            await s.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rag_vectors_embedding "
                    "ON rag_vectors USING hnsw (embedding vector_cosine_ops)"
                )
            )
            await s.commit()
        self._table_ready = True

    async def ensure_collection(
        self,
        collection: str,
        *,
        vector_size: int,
        embedding_model: str,
        distance: str = "cosine",
    ) -> None:
        if vector_size != _DIM:
            raise ValueError(
                f"PgVectorStore is fixed at dim {_DIM} (migration 0009), but the "
                f"embedding model produced dim {vector_size}. Re-run a migration "
                f"to change the vector column dimension, or use a {_DIM}-dim model."
            )
        await self._ensure_table()  # idempotent; creates ext+table if needed
        self._ensured.add(collection)  # collections are shared in one table

    async def has_collection(self, collection: str) -> bool:
        async with self._factory()() as s:
            r = await s.execute(
                text("SELECT 1 FROM rag_vectors WHERE collection = :c LIMIT 1"),
                {"c": collection},
            )
            return r.first() is not None

    async def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        if not ids:
            return
        async with self._factory()() as s:
            for pid, vec, payload in zip(ids, vectors, payloads):
                await s.execute(
                    text(
                        """
                        INSERT INTO rag_vectors (id, collection, content, payload, embedding)
                        VALUES (:id, :c, :content, CAST(:payload AS jsonb), CAST(:emb AS vector))
                        ON CONFLICT (id) DO UPDATE
                          SET content = EXCLUDED.content,
                              payload = EXCLUDED.payload,
                              embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "id": _coerce_uuid(pid),
                        "c": collection,
                        "content": (payload or {}).get("content", ""),
                        "payload": json.dumps(payload or {}),
                        "emb": _vec_literal(vec),
                    },
                )
            await s.commit()

    async def query(
        self,
        collection: str,
        *,
        vector: list[float],
        k: int = 10,
        filter: dict | None = None,  # noqa: A002 — protocol name
    ) -> list[Hit]:
        async with self._factory()() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT id, content, payload,
                               1 - (embedding <=> CAST(:q AS vector)) AS score
                        FROM rag_vectors
                        WHERE collection = :c
                        ORDER BY embedding <=> CAST(:q AS vector)
                        LIMIT :k
                        """
                    ),
                    {"q": _vec_literal(vector), "c": collection, "k": k},
                )
            ).mappings().all()
        return [_row_to_hit(r) for r in rows]

    async def _bm25(self, collection: str, query_text: str, k: int) -> list[Hit]:
        if not (query_text or "").strip():
            return []
        async with self._factory()() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT id, content, payload,
                               ts_rank(content_tsv,
                                       websearch_to_tsquery('english', :q)) AS score
                        FROM rag_vectors
                        WHERE collection = :c
                          AND content_tsv @@ websearch_to_tsquery('english', :q)
                        ORDER BY score DESC
                        LIMIT :k
                        """
                    ),
                    {"q": query_text, "c": collection, "k": k},
                )
            ).mappings().all()
        return [_row_to_hit(r) for r in rows]

    async def hybrid_query(
        self,
        collection: str,
        *,
        vector: list[float],
        query_text: str,
        k: int = 10,
        k_each: int = 30,
    ) -> list[Hit]:
        """Dense ANN + BM25, fused by Reciprocal Rank Fusion."""
        dense = await self.query(collection, vector=vector, k=k_each)
        sparse = await self._bm25(collection, query_text, k_each)
        return _rrf([dense, sparse])[:k]

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filter: dict | None = None,  # noqa: A002
    ) -> None:
        async with self._factory()() as s:
            if ids:
                await s.execute(
                    text(
                        "DELETE FROM rag_vectors WHERE collection = :c "
                        "AND id = ANY(:ids)"
                    ),
                    {"c": collection, "ids": [_coerce_uuid(i) for i in ids]},
                )
            else:
                await s.execute(
                    text("DELETE FROM rag_vectors WHERE collection = :c"),
                    {"c": collection},
                )
            await s.commit()

    async def reset(self, collection: str) -> None:
        """Drop every vector in `collection`. Required by the VectorStore
        protocol — its absence meant resume re-uploads and conversation
        deletes silently LEAKED vectors (the AttributeError was swallowed by
        fail-open callers), diluting retrieval with stale chunks (audit
        2026-07-09)."""
        await self.delete(collection)

    async def close(self) -> None:  # pragma: no cover — pool owned by storage.db
        return None


def _coerce_uuid(pid: str) -> str:
    """Accept hex / UUID strings; normalize to canonical UUID text."""
    try:
        return str(uuid.UUID(str(pid)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_OID, str(pid)))


def _row_to_hit(r) -> Hit:
    payload = _as_payload(r["payload"])
    content = r["content"] or payload.get("content", "")
    return Hit(
        id=str(r["id"]),
        score=float(r["score"]),
        payload=payload,
        document=content,
    )


def _rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion across several ranked lists."""
    scores: dict[str, float] = {}
    items: dict[str, Hit] = {}
    for ranked in rankings:
        for rank, hit in enumerate(ranked):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            items.setdefault(hit.id, hit)
    fused = sorted(items.values(), key=lambda h: -scores[h.id])
    for h in fused:
        h.score = scores[h.id]
    return fused
