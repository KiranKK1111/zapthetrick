"""RAG for chat attachments — persistent (Qdrant) + a stateless fallback.

Every uploaded document is ingested into a **per-conversation** vector
collection (`chat_docs_{conversation_id}`) so its chunks persist and are
reused: the upload turn and any later follow-up (via the Retriever agent)
retrieve the relevant chunks for the question. Reuses the existing
[chunker](app/rag/chunker.py), [embedder](app/rag/embedder.py), and the
backend-agnostic vector store ([storage.vectors]).

`select_relevant_chunks` is a stateless fallback (chunk+embed+cosine in
memory) used when the vector store is unavailable.
"""
from __future__ import annotations

import asyncio
import uuid

from app.core.config_loader import cfg
from app.rag import embedder
from app.rag.chunker import chunk_resume


# Safety net on background embedding: cap how many chunks one document embeds
# so a very large file can't tie up the CPU embedder indefinitely. Paired with
# the extraction cap in documents/parser.py (MAX_EXTRACT_CHARS), this bounds a
# 100 MB upload's ingest to minutes, not hours.
MAX_INGEST_CHUNKS = 6000


def _collection(conversation_id: str) -> str:
    return f"chat_docs_{conversation_id}"


async def ingest_chat_document(
    conversation_id: str,
    filename: str,
    text: str,
) -> int:
    """Chunk + embed `text` and upsert into the conversation's vector
    collection. Returns the number of chunks stored (0 on failure / empty)."""
    chunks = chunk_resume(
        text, chunk_size=cfg.rag.chunk_size, chunk_overlap=cfg.rag.chunk_overlap
    )
    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        return 0
    if len(texts) > MAX_INGEST_CHUNKS:
        texts = texts[:MAX_INGEST_CHUNKS]
    try:
        from storage.vectors import get_vector_store

        embeddings = await asyncio.to_thread(embedder.embed, texts)
        store = get_vector_store()
        collection = _collection(conversation_id)
        await store.ensure_collection(
            collection,
            vector_size=len(embeddings[0]),
            embedding_model=cfg.embeddings.model,
        )
        await store.upsert(
            collection,
            ids=[uuid.uuid4().hex for _ in texts],
            vectors=embeddings,
            payloads=[
                {"content": texts[i], "filename": filename, "position": i}
                for i in range(len(texts))
            ],
        )
        # Content KG (Architecture §3.1): extract entities/relations at INGEST
        # (one call, off the hot path) and persist per-conversation, so per-turn
        # related-concept suggestions are a cheap LOCAL query (no per-turn LLM).
        if getattr(cfg.advanced_rag, "use_knowledge_graph", False):
            try:
                await _persist_conversation_kg(conversation_id, text)
            except Exception:  # noqa: BLE001 — KG is additive, never blocks ingest
                pass
        # Document vision (Phase 3 #21): layout-aware + chart understanding over
        # the extracted markdown — heading skeleton, table/figure inventory, and
        # a per-table chart summary (min/max/trend). Persisted per-conversation
        # so a later turn can reference the document's structure. Additive +
        # fail-open — never blocks ingest.
        if getattr(cfg.documents, "doc_vision", True):
            try:
                await _persist_doc_layout(conversation_id, text)
            except Exception:  # noqa: BLE001 — vision layer is additive
                pass
        return len(texts)
    except Exception:  # noqa: BLE001 — vector store offline: skip persistence
        return 0


async def _persist_conversation_kg(conversation_id: str, text: str) -> None:
    """Extract the doc's entity/relation graph and merge it into the persisted
    KG. §17: when the conversation belongs to a project, the KG accretes at the
    PROJECT level (`Project.metadata['kg']`) so every chat in the project shares
    it; otherwise it stays per-conversation (`Session.metadata['kg']`).
    Best-effort."""
    from app.rag.kg_extract import extract_graph, merge_json
    nodes, edges = await extract_graph(text)
    if not nodes:
        return
    from storage.db import get_session_factory
    from storage.models import Project as _Project, Session as _SessionRow
    f = get_session_factory()
    if f is None:
        return
    async with f() as ws:
        row = await ws.get(_SessionRow, uuid.UUID(str(conversation_id)))
        if row is None:
            return
        target = row
        pid = getattr(row, "project_id", None)
        if pid is not None:
            proj = await ws.get(_Project, pid)
            if proj is not None:
                target = proj
        meta = dict(_kg_meta(target) or {})
        meta["kg"] = merge_json(meta.get("kg"), nodes, edges)
        _set_kg_meta(target, meta)    # reassign so SQLAlchemy flags the change
        await ws.commit()


async def _persist_doc_layout(conversation_id: str, text: str) -> None:
    """Analyze the ingested document's layout + charts and store a compact
    summary in the conversation's (or project's) metadata. Best-effort."""
    from app.rag.doc_vision import analyze_layout
    layout = analyze_layout(text)
    # Nothing structural found → don't bloat metadata with an empty record.
    if not (layout.headings or layout.tables or layout.charts):
        return
    from storage.db import get_session_factory
    from storage.models import Project as _Project, Session as _SessionRow
    f = get_session_factory()
    if f is None:
        return
    async with f() as ws:
        row = await ws.get(_SessionRow, uuid.UUID(str(conversation_id)))
        if row is None:
            return
        target = row
        pid = getattr(row, "project_id", None)
        if pid is not None:
            proj = await ws.get(_Project, pid)
            if proj is not None:
                target = proj
        meta = dict(_kg_meta(target) or {})
        meta["doc_layout"] = layout.as_dict()
        _set_kg_meta(target, meta)
        await ws.commit()


async def load_conversation_kg(conversation_id: str):
    """The persisted content KG for a conversation — the PROJECT graph when the
    conversation belongs to one (§17), else the per-conversation graph. Returns
    a `KnowledgeGraph` or None when absent/unavailable (fail-open)."""
    from app.rag.kg_extract import graph_from_json
    from storage.db import get_session_factory
    from storage.models import Project as _Project, Session as _SessionRow
    f = get_session_factory()
    if f is None:
        return None
    try:
        async with f() as ws:
            row = await ws.get(_SessionRow, uuid.UUID(str(conversation_id)))
            if row is None:
                return None
            data = None
            pid = getattr(row, "project_id", None)
            if pid is not None:
                proj = await ws.get(_Project, pid)
                if proj is not None and isinstance(
                        getattr(proj, "project_metadata", None), dict):
                    data = proj.project_metadata.get("kg")
            if data is None and isinstance(
                    getattr(row, "session_metadata", None), dict):
                data = row.session_metadata.get("kg")
            if not data:
                return None
            return graph_from_json(data)
    except Exception:  # noqa: BLE001 — KG is additive, never blocks a turn
        return None


def _kg_meta(row) -> dict:
    """The metadata dict of a Session or Project row (both name the column
    `metadata`, exposed as `session_metadata` / `project_metadata`)."""
    return getattr(row, "project_metadata", None) if hasattr(
        row, "project_metadata") else getattr(row, "session_metadata", None)


def _set_kg_meta(row, meta: dict) -> None:
    if hasattr(row, "project_metadata"):
        row.project_metadata = meta
    else:
        row.session_metadata = meta


async def drop_chat_collection(conversation_id: str) -> None:
    """Remove a conversation's chat-document vectors (called on chat delete)."""
    try:
        from storage.vectors import get_vector_store

        await get_vector_store().reset(_collection(conversation_id))
    except Exception:  # noqa: BLE001 — vector store offline / collection absent
        pass


async def retrieve_chat_hits(
    conversation_id: str,
    query: str,
    *,
    k: int = 8,
) -> list[dict]:
    """Strong retrieval for `query` over the conversation's documents.

    Pipeline: HyDE query expansion → hybrid (dense ANN + BM25, RRF-fused when
    the store supports it) → cross-encoder rerank → top-k. Each stage degrades
    gracefully, so a missing reranker / LLM just yields a slightly weaker
    ranking rather than failing. Returns [{content, filename, score}].
    """
    if not query.strip():
        return []
    try:
        from app.rag.query_expand import hyde_text
        from storage.vectors import get_vector_store

        # 1. Query expansion (HyDE) → the text we embed for DENSE search. BM25
        #    keeps the literal query so exact terms still match.
        dense_text = await hyde_text(query)
        qv = await asyncio.to_thread(embedder.embed_one, dense_text)

        store = get_vector_store()
        collection = _collection(conversation_id)
        await store.ensure_collection(
            collection, vector_size=len(qv), embedding_model=cfg.embeddings.model
        )

        # 2. Hybrid (dense + BM25) when the backend supports it; else dense.
        wide = max(k * 4, 24)
        if hasattr(store, "hybrid_query"):
            hits = await store.hybrid_query(
                collection, vector=qv, query_text=query, k=wide
            )
        else:
            hits = await store.query(collection, vector=qv, k=wide)
    except Exception:  # noqa: BLE001 — no collection yet / store offline
        return []

    candidates: list[dict] = []
    for h in hits:
        payload = h.payload or {}
        candidates.append(
            {
                "content": h.document or payload.get("content", ""),
                "filename": payload.get("filename", "document"),
                "score": float(h.score),
            }
        )
    if not candidates:
        return []

    # 3. Cross-encoder rerank the wide candidate set down to a pool. With MMR on
    #    we keep a WIDER pool (2k) so the diversity pass has something to choose
    #    between; with it off the pool is k and the behaviour is unchanged.
    use_mmr = _use_mmr()
    pool_k = k * 2 if use_mmr else k
    try:
        from app.rag.rerank import rerank_hits

        ranked = await asyncio.to_thread(rerank_hits, query, candidates, pool_k)
    except Exception:  # noqa: BLE001 — rerank is best-effort
        ranked = candidates[:pool_k]

    # 4. MMR diversity pass (`advanced_rag.use_mmr`): a chunked document
    #    produces overlapping windows, so the top-k by relevance is often the
    #    same passage k times. Fail-open to the reranked order.
    if use_mmr and len(ranked) > k:
        try:
            from app.rag.mmr import DEFAULT_LAMBDA, mmr_filter

            picked = mmr_filter(
                query,
                ranked,
                top_k=k,
                lambda_=float(
                    getattr(cfg.advanced_rag, "mmr_lambda", DEFAULT_LAMBDA)
                ),
            )
            if picked:
                return picked
        except Exception:  # noqa: BLE001 — diversity is best-effort
            pass
    return ranked[:k]


def _use_mmr() -> bool:
    return bool(getattr(cfg.advanced_rag, "use_mmr", False))


async def retrieve_chat_context(conversation_id: str, query: str, *, k: int = 8) -> str:
    """Retrieved chunks formatted as a labelled context block, or ''."""
    hits = await retrieve_chat_hits(conversation_id, query, k=k)
    return "\n\n".join(f"[{h['filename']}] {h['content']}" for h in hits)


async def select_relevant_chunks(
    text: str,
    query: str,
    *,
    max_chunks: int = 12,
) -> str:
    """Return the chunks of `text` most relevant to `query`, joined.

    If the document chunks to <= max_chunks we return them all (no point
    embedding). Otherwise we embed everything and keep the top-k by cosine
    similarity, restored to document order so the model reads them coherently.
    Falls back to a head-truncation if embedding is unavailable.

    When `advanced_rag.use_mmr` is on, the top-k selection is a TRUE cosine MMR:
    this function has already embedded every chunk, so diversity costs nothing
    extra here — we reuse those vectors rather than re-embedding.
    """
    chunks = chunk_resume(
        text,
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
    )
    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        return text[: cfg.rag.chunk_size * max_chunks]
    if len(texts) <= max_chunks:
        return "\n\n".join(texts)

    try:
        import numpy as np

        embs = await asyncio.to_thread(embedder.embed, texts)
        qv = await asyncio.to_thread(embedder.embed_one, query)
        matrix = np.array(embs, dtype="float32")
        q = np.array(qv, dtype="float32")
        scores = matrix @ q  # embeddings are unit-normalized → dot == cosine

        top: list[int] | None = None
        if _use_mmr():
            try:
                from app.rag.mmr import DEFAULT_LAMBDA, mmr_select

                top = mmr_select(
                    [float(s) for s in scores],
                    vectors=embs,          # already computed above — free
                    top_k=max_chunks,
                    lambda_=float(
                        getattr(cfg.advanced_rag, "mmr_lambda", DEFAULT_LAMBDA)
                    ),
                )
            except Exception:  # noqa: BLE001 — diversity is best-effort
                top = None
        if not top:
            top = [int(i) for i in np.argsort(-scores)[:max_chunks]]

        for_doc_order = sorted(int(i) for i in top)
        return "\n\n".join(texts[i] for i in for_doc_order)
    except Exception:  # noqa: BLE001 — embeddings unavailable: head-truncate
        return "\n\n".join(texts[:max_chunks])
