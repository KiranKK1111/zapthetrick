"""Resume ingest pipeline: text -> chunks -> embeddings -> Postgres + VectorStore.

  - Chunks land in Postgres via [ResumeRepo.replace_chunks] (idempotent
    on `resume_id`: existing chunks for this resume are wiped first).
  - Embeddings land in the configured [VectorStore] (Qdrant by default,
    Chroma alternative), in a per-resume collection so each upload
    gets its own HNSW index.
  - The chunk rows carry a `vector_point_id` UUID so Postgres -> vector
    pointers stay tight; rebuilding the vector index reads the rows back
    and re-upserts.
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_loader import cfg
from app.documents import progress as _progress
from app.rag import embedder, retriever, store
from app.rag.chunker import chunk_resume
from storage.repos import ResumeRepo

# Embed in batches so progress can tick within the stage ("Embedding
# chunks 42/120") instead of one opaque multi-second call.
_EMBED_BATCH = 16


class IngestWarning(RuntimeError):
    """Non-fatal: chunks saved in Postgres but the vector store is degraded."""


async def ingest_resume(
    resume_id: str,
    raw_text: str,
    session: AsyncSession,
) -> int:
    """Chunk, embed, persist. Returns the number of chunks stored.

    Idempotent on `resume_id`: rewrites the chunks + the vector
    collection from scratch each call.
    """
    # 1. Drop any previous vectors for this resume.
    try:
        await store.delete_resume(resume_id)
    except Exception:
        # Vector store offline; soldier on — Postgres is authoritative.
        pass
    retriever.invalidate_bm25(resume_id)

    # Progress: chunking stage. Fail-open registry — resume_id may or may
    # not have been `begin()`-ed (reindex/tests call this directly).
    _progress.set_stage(resume_id, "chunk", detail="Splitting resume into sections")

    chunks = chunk_resume(
        raw_text,
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
    )
    if not chunks:
        _progress.update(resume_id, fraction=1.0, detail="No chunkable text found")
        return 0
    _progress.update(
        resume_id,
        fraction=1.0,
        detail=f"Split into {len(chunks)} chunks",
        counts={"chunks": len(chunks)},
    )

    # 2. Pre-mint vector_point_id values so the Postgres row + vector
    #    payload reference the same UUID. That makes citation panels
    #    cheap (resolve vector hit -> Postgres row by point id).
    chunk_payloads: list[dict] = []
    for ch in chunks:
        chunk_payloads.append(
            {
                "content": ch.text,
                "section_type": ch.section,
                "level": 1,                # Phase-1 chunker is flat; hierarchical chunker fills in deeper levels.
                "position": ch.position,
                "vector_point_id": uuid.uuid4(),
            }
        )

    # 3. Insert chunks into Postgres (durable, queryable, FTS-indexed).
    repo = ResumeRepo(session)
    rows = await repo.replace_chunks(resume_id, chunk_payloads)
    if not rows:
        return 0

    # 4. Embed and upsert into the VectorStore. Batched so the progress
    #    registry can tick within the stage.
    #
    # Contextual Retrieval (P3 #19): embed the CONTEXTUALIZED text — a short
    # doc/section header prepended to each chunk — so an otherwise-orphaned line
    # ("improved throughput 40%") carries what it's about. Only the EMBEDDING
    # input is contextualized; Postgres + the vector payload keep the raw
    # content, so retrieval still returns the original text. Gated + fail-open.
    if getattr(cfg.rag, "use_contextual_compression", True):
        from app.rag import contextual
        texts = [
            contextual.contextualize(
                r.content,
                doc_title="Resume",
                section=(getattr(r, "section_type", "") or ""),
            )
            for r in rows
        ]
    else:
        texts = [r.content for r in rows]
    total = len(texts)
    _progress.set_stage(
        resume_id,
        "embed",
        detail=f"Embedding chunks 0/{total}",
        counts={"chunks": total, "embedded": 0},
    )
    embeddings: list[list[float]] = []
    for start in range(0, total, _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        embeddings.extend(await asyncio.to_thread(embedder.embed, batch))
        done = min(start + _EMBED_BATCH, total)
        _progress.update(
            resume_id,
            fraction=done / total,
            detail=f"Embedding chunks {done}/{total}",
            counts={"embedded": done},
        )
    ids = [str(r.vector_point_id) for r in rows]
    documents = texts
    metadatas = [
        {
            "resume_id": str(r.resume_id),
            "chunk_id": str(r.id),
            "position": r.position,
            "section": r.section_type or "",
        }
        for r in rows
    ]
    _progress.set_stage(
        resume_id, "index", detail=f"Indexing {total} vectors"
    )
    try:
        await store.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:  # noqa: BLE001
        # Postgres has the authoritative copy; the retriever falls back
        # to BM25-only when vectors are unavailable.
        raise IngestWarning(
            f"Chunks stored in Postgres but vector upsert failed: {exc}"
        ) from exc
    _progress.update(
        resume_id, fraction=1.0, counts={"indexed": total},
        detail=f"Indexed {total} vectors",
    )

    return len(rows)
