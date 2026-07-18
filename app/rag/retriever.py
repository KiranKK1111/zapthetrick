"""Hybrid retriever: BM25 (Postgres FTS) + vector (Qdrant/Chroma) + RRF + rerank.

Why hybrid:
  - BM25 catches keyword-heavy queries ("Did you use Kafka?").
  - Vector catches paraphrased queries ("Tell me about your distributed
    systems work").

Reciprocal Rank Fusion merges the two without weight tuning. A
cross-encoder rerank on the top fused candidates gives a calibrated
final ordering (single biggest quality win in resume RAG).

Storage:
  - Vectors: per-resume collection in [VectorStore] (Qdrant/Chroma).
  - Keyword: Postgres `content_tsv` GIN index via [ResumeRepo.search_chunks_fts].
  - Source-of-truth chunks: `resume_chunks` in Postgres.

Pointer convention: each ResumeChunk row has `id` (Postgres PK) and
`vector_point_id` (the vector store's ID). Vector hits come back as
qdrant point ids; we resolve them to chunks by joining on that field.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_loader import cfg
from app.rag import embedder, store
from app.rag.mmr import DEFAULT_LAMBDA, mmr_filter
from storage.repos import ResumeRepo


log = logging.getLogger(__name__)

# How many reranked candidates the MMR diversity pass gets to choose from.
# Bounded on purpose: MMR is O(k·n) similarity comparisons, and a wider pool
# just drags low-relevance chunks into contention.
MMR_POOL = 20


class RetrieverError(RuntimeError):
    pass


@dataclass
class RetrievalHit:
    """One chunk plus the score it received from the final ranker."""
    chunk_id: str
    text: str
    section: str | None
    position: int
    score: float


# Cached BM25 used to live here. With Postgres FTS the index is server-
# side, so `invalidate_bm25` is a kept-for-compat no-op.
def invalidate_bm25(resume_id: str) -> None:
    """No-op — Postgres FTS index updates automatically.

    Kept so legacy ingest code that calls this on re-ingest keeps
    compiling. Remove once every call site is gone.
    """
    return None


# ---- Reranker ----------------------------------------------------------
@lru_cache(maxsize=1)
def _reranker():
    if not cfg.reranker.enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RetrieverError(
            "sentence-transformers is not installed for the reranker."
        ) from exc
    # local-first: cached model skips the online HEAD check; un-cached → fetch.
    try:
        return CrossEncoder(cfg.reranker.model, local_files_only=True)
    except Exception:  # noqa: BLE001
        return CrossEncoder(cfg.reranker.model)


# ---- Public API --------------------------------------------------------
async def retrieve(
    query: str,
    *,
    resume_id: str,
    session: AsyncSession,
    section: str | None = None,
) -> list[RetrievalHit]:
    """Top-k hits for `query` against the given resume, hybrid + reranked.

    Steps:
      1. Vector search via [VectorStore.query].
      2. BM25 keyword search via Postgres `content_tsv` GIN index.
      3. Reciprocal Rank Fusion.
      4. Cross-encoder rerank on the top-20 fused candidates.
      5. MMR diversity pass over the reranked pool (`advanced_rag.use_mmr`)
         so the final set isn't k paraphrases of one chunk.
      6. Return `cfg.rag.top_k_rerank` final hits.
    """
    if not query.strip():
        return []

    repo = ResumeRepo(session)

    # Vector side. Off-load embedding to a thread (CPU-bound).
    q_emb = await asyncio.to_thread(embedder.embed_one, query)
    vector_hits = await store.query(
        q_emb, k=cfg.rag.top_k_retrieve, resume_id=resume_id, section=section
    )
    # Vector hit ids are qdrant point ids — resolve to chunk_id via metadata.
    vec_chunk_ids: list[str] = []
    for h in vector_hits:
        meta = h.get("metadata") or {}
        cid = meta.get("chunk_id")
        if cid:
            vec_chunk_ids.append(str(cid))

    # BM25 side — Postgres FTS.
    fts_rows = await repo.search_chunks_fts(
        resume_id, query, limit=cfg.rag.top_k_retrieve
    )
    bm25_chunk_ids = [str(r.id) for r in fts_rows]

    if not vec_chunk_ids and not bm25_chunk_ids:
        return []

    # Fuse.
    fused_ids = _rrf(vec_chunk_ids, bm25_chunk_ids)

    # Load all chunks we'll need by ID. Use the rows we already fetched
    # for BM25 + a top-up fetch for vector-only matches.
    chunk_by_id = {str(r.id): r for r in fts_rows}
    missing_ids = [cid for cid in fused_ids if cid not in chunk_by_id]
    if missing_ids:
        # We don't have a bulk-by-id repo method; fetch the resume's
        # chunks and filter. For a single resume that's hundreds of
        # rows max — cheap.
        all_chunks = await repo.fetch_chunks(resume_id)
        for c in all_chunks:
            chunk_by_id.setdefault(str(c.id), c)

    # Rerank the top-20 fused candidates.
    reranker = _reranker() if cfg.reranker.enabled else None
    if reranker is not None:
        candidate_ids = fused_ids[:20]
        pairs: list[tuple[str, str]] = []
        valid_ids: list[str] = []
        for cid in candidate_ids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue
            pairs.append((query, chunk.content))
            valid_ids.append(cid)
        if pairs:
            rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
            order = sorted(
                range(len(valid_ids)),
                key=lambda i: float(rerank_scores[i]),
                reverse=True,
            )
            final_ids = [valid_ids[i] for i in order]
            final_scores = {
                valid_ids[i]: float(rerank_scores[i]) for i in range(len(valid_ids))
            }
        else:
            final_ids = fused_ids
            final_scores = {cid: 1.0 / (i + 1) for i, cid in enumerate(fused_ids)}
    else:
        final_ids = fused_ids
        final_scores = {cid: 1.0 / (i + 1) for i, cid in enumerate(fused_ids)}

    top_k = cfg.rag.top_k_rerank

    # Observability (health dashboard, P1 #6): record the mean top-k relevance
    # so the dashboard can report retrieval quality. Fail-open — never affects
    # what is returned.
    try:
        from app.obs.metrics import record_retrieval_relevance
        _top_scores = [float(final_scores.get(cid, 0.0))
                       for cid in final_ids[:top_k]]
        if _top_scores:
            record_retrieval_relevance(sum(_top_scores) / len(_top_scores))
    except Exception:  # noqa: BLE001
        pass

    def _hits_for(ids: list[str]) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for cid in ids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=str(chunk.id),
                    text=chunk.content,
                    section=chunk.section_type,
                    position=chunk.position,
                    score=final_scores.get(cid, 0.0),
                )
            )
        return hits

    # 5. Diversity pass. Relevance-ranking alone happily returns k
    #    near-identical chunks (overlapping chunk windows, parent/child
    #    hierarchical chunks) — MMR picks the top-k that COVER the query
    #    instead. Honours `advanced_rag.use_mmr`, which was set but never
    #    read by any retriever until now (audit 2026-07-14).
    #
    #    Fail-open: anything unexpected here and we return exactly what the
    #    pipeline returned before MMR existed.
    if bool(getattr(cfg.advanced_rag, "use_mmr", False)):
        try:
            pool = _hits_for(final_ids[:MMR_POOL])
            if len(pool) > top_k:
                picked = mmr_filter(
                    query,
                    pool,
                    top_k=top_k,
                    lambda_=float(
                        getattr(cfg.advanced_rag, "mmr_lambda", DEFAULT_LAMBDA)
                    ),
                )
                if picked:
                    return picked
        except Exception as exc:  # noqa: BLE001 — diversity is best-effort
            log.warning("MMR pass failed, keeping the reranked order: %s", exc)

    return _hits_for(final_ids[:top_k])


def _rrf(*ranked_lists: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. `k=60` is the original-paper constant —
    works well out of the box."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
