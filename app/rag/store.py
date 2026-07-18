"""Vector store shim — delegates to [storage.vectors] factory.

Used to be a thin Chroma wrapper. Now routes through the backend-
agnostic [VectorStore], so flipping `cfg.vector_store.provider` from
`chroma` to `qdrant` (or any future adapter) doesn't require touching
any caller.

Collection naming follows DataBaseArchitecture.md: one collection per
resume (`resume_chunks_{resume_id}`). Better isolation than a shared
collection — HNSW perf stays good per-resume, and deleting a resume
is a single `delete_collection` call.

All functions are async now. Old sync call sites have been ported
to await them; the supervisor / ingest path was already inside the
event loop anyway.
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.config_loader import cfg
from storage.vectors import get_vector_store


log = logging.getLogger(__name__)


def _collection(resume_id: str) -> str:
    return f"resume_chunks_{resume_id}"


def _vector_size_default() -> int:
    # bge-small produces 384-dim vectors; bge-m3 dense is 1024.
    name = (cfg.embeddings.model or "").lower()
    if "m3" in name:
        return 1024
    return 384


async def upsert(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Group rows by `metadata.resume_id` and upsert into per-resume
    collections. Every metadata dict must carry `resume_id`."""
    if not ids:
        return

    by_resume: dict[str, dict[str, list]] = {}
    for i, _id in enumerate(ids):
        rid = (metadatas[i] or {}).get("resume_id")
        if not rid:
            raise ValueError(
                f"vector upsert at index {i} is missing metadata.resume_id"
            )
        bucket = by_resume.setdefault(
            rid, {"ids": [], "vectors": [], "payloads": []}
        )
        bucket["ids"].append(_id)
        bucket["vectors"].append(embeddings[i])
        # Stash the document on the payload so the citation pane and
        # rerankers don't need a Postgres roundtrip.
        payload = {**(metadatas[i] or {}), "content": documents[i]}
        bucket["payloads"].append(payload)

    store = get_vector_store()
    for rid, bucket in by_resume.items():
        collection = _collection(rid)
        vector_size = len(bucket["vectors"][0]) if bucket["vectors"] else _vector_size_default()
        await store.ensure_collection(
            collection,
            vector_size=vector_size,
            embedding_model=cfg.embeddings.model,
        )
        await store.upsert(collection, **bucket)


async def query(
    query_embedding: list[float],
    *,
    k: int,
    resume_id: str | None = None,
    section: str | None = None,
) -> list[dict]:
    """Returns `[{id, document, metadata, distance}]` — same shape the
    old Chroma adapter produced, so legacy retrievers keep working."""
    if resume_id is None:
        return []

    store = get_vector_store()
    collection = _collection(resume_id)
    flt = {"section": section} if section else None
    try:
        await store.ensure_collection(
            collection,
            vector_size=len(query_embedding),
            embedding_model=cfg.embeddings.model,
        )
    except Exception:
        # Collection doesn't exist yet (no ingest has run). Return [].
        return []

    hits = await store.query(collection, vector=query_embedding, k=k, filter=flt)
    return [
        {
            "id": h.id,
            "document": h.document or h.payload.get("content", ""),
            "metadata": h.payload,
            # Distance = 1 - score for cosine — the old shape.
            "distance": 1.0 - float(h.score),
        }
        for h in hits
    ]


async def delete_resume(resume_id: str) -> None:
    """Drop everything for one resume — used by `ingest_resume` on re-upload."""
    store = get_vector_store()
    try:
        await store.reset(_collection(resume_id))
    except Exception as exc:
        log.warning("delete_resume(%s) ignored: %s", resume_id, exc)
