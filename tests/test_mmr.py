"""MMR diversity pass (`advanced_rag.use_mmr`).

Two bugs this suite locks down:

  1. `mmr_filter` existed, the flag was ON, and NO retriever imported it — so
     retrieval happily returned k near-identical chunks. These tests assert the
     duplicates are actually dropped, and that the wiring in `app.rag.retriever`
     really runs.
  2. MMR must be strictly additive: with the flag off, or when MMR blows up,
     retrieval returns exactly what it returned before MMR existed.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import pytest

from app.core.config_loader import cfg
from app.rag import documents, retriever
from app.rag.mmr import cosine, mmr_filter, mmr_select, token_cosine


# ---- fixtures / fakes ---------------------------------------------------
@dataclass
class _Hit:
    """Shaped like RetrievalHit (what the retriever hands MMR)."""
    text: str
    score: float


@dataclass
class _Chunk:
    """Shaped like a `resume_chunks` row."""
    id: uuid.UUID
    content: str
    section_type: str | None = "experience"
    position: int = 0


# Three near-duplicates (the same sentence with trivial edits — exactly what
# `rag.chunk_overlap` and hierarchical parent/child chunking produce) plus two
# genuinely different chunks.
DUP_A = "Built a Kafka pipeline processing 2M events per day at Acme."
DUP_B = "Built a Kafka pipeline processing 2M events per day at Acme Corp."
DUP_C = "At Acme, built a Kafka pipeline processing 2M events per day."
DISTINCT_1 = "Led the migration of the billing service from MySQL to Postgres."
DISTINCT_2 = "Mentored four junior engineers and ran the hiring loop."


def _near_dup_hits() -> list[_Hit]:
    # Descending relevance — the three duplicates outrank both distinct chunks,
    # so a pure relevance ranking returns nothing but paraphrases.
    return [
        _Hit(DUP_A, 9.0),
        _Hit(DUP_B, 8.5),
        _Hit(DUP_C, 8.2),
        _Hit(DISTINCT_1, 4.0),
        _Hit(DISTINCT_2, 3.0),
    ]


# ---- similarity primitives ---------------------------------------------
def test_token_cosine_separates_near_dupes_from_distinct_chunks():
    assert token_cosine(DUP_A, DUP_B) > 0.85          # near-identical
    assert token_cosine(DUP_A, DUP_C) > 0.85
    assert token_cosine(DUP_A, DISTINCT_1) < 0.25     # unrelated evidence
    assert token_cosine(DUP_A, "") == 0.0


def test_cosine_handles_orthogonal_and_identical_vectors():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0      # zero vector, no NaN


# ---- the filter itself --------------------------------------------------
def test_mmr_drops_near_duplicates():
    """THE bug: without MMR the top-3 is three paraphrases of one chunk."""
    hits = _near_dup_hits()

    baseline = [h.text for h in hits[:3]]             # what relevance alone gives
    assert baseline == [DUP_A, DUP_B, DUP_C]          # …all the same evidence

    picked = [h.text for h in mmr_filter("kafka work at acme", hits, top_k=3)]

    assert picked[0] == DUP_A                         # most relevant still first
    assert len(picked) == 3
    # At most ONE of the three near-duplicates survives.
    assert sum(t in (DUP_A, DUP_B, DUP_C) for t in picked) == 1
    # …and the slots it freed went to the distinct evidence.
    assert DISTINCT_1 in picked and DISTINCT_2 in picked


def test_mmr_drops_exact_duplicates():
    hits = [_Hit(DUP_A, 9.0), _Hit(DUP_A, 8.9), _Hit(DISTINCT_1, 1.0)]
    picked = mmr_filter("q", hits, top_k=3)
    assert [h.text for h in picked] == [DUP_A, DISTINCT_1]


def test_mmr_uses_supplied_vectors_when_available():
    """The `select_relevant_chunks` path: vectors already computed → true
    cosine MMR, no re-embedding. Text here is deliberately dissimilar so only
    the VECTORS can drive the dedupe."""
    hits = [_Hit("alpha", 9.0), _Hit("beta", 8.0), _Hit("gamma", 1.0)]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],   # a near-duplicate of #0 in embedding space
        [0.0, 0.0, 1.0],     # orthogonal → genuinely new evidence
    ]
    picked = [h.text for h in mmr_filter("q", hits, top_k=2, vectors=vectors)]
    assert picked == ["alpha", "gamma"]


def test_mmr_reads_dict_hits_and_rerank_scores():
    """The chat-document path hands MMR dicts, not dataclasses."""
    hits = [
        {"content": DUP_A, "score": 0.1, "rerank_score": 9.0},
        {"content": DUP_B, "score": 0.9, "rerank_score": 8.0},
        {"content": DISTINCT_1, "score": 0.2, "rerank_score": 1.0},
    ]
    picked = [h["content"] for h in mmr_filter("q", hits, top_k=2)]
    assert picked == [DUP_A, DISTINCT_1]   # rerank_score wins, dup dropped


def test_mmr_never_invents_or_duplicates_hits():
    hits = _near_dup_hits()
    picked = mmr_filter("q", hits, top_k=10)
    assert all(p in hits for p in picked)              # same objects
    assert len({id(p) for p in picked}) == len(picked)  # no repeats


def test_mmr_handles_empty_and_singleton_input():
    assert mmr_filter("q", [], top_k=5) == []
    one = [_Hit(DUP_A, 1.0)]
    assert mmr_filter("q", one, top_k=5) == one
    assert mmr_filter("q", one, top_k=0) == []


def test_mmr_select_degenerates_to_top_k_with_lambda_one():
    """λ=1 means 'pure relevance' — MMR must then reproduce the input ranking."""
    rel = [0.9, 0.8, 0.7, 0.6]
    texts = [DUP_A, DUP_B, DUP_C, DISTINCT_1]
    assert mmr_select(rel, texts=texts, top_k=3, lambda_=1.0) == [0, 1, 2]


def test_mmr_select_with_equal_scores_still_diversifies():
    rel = [1.0, 1.0, 1.0]
    texts = [DUP_A, DUP_B, DISTINCT_1]
    picked = mmr_select(rel, texts=texts, top_k=2, lambda_=0.5)
    assert picked == [0, 2]          # not [0, 1] — the paraphrase is skipped


# ---- retriever wiring ---------------------------------------------------
class _FakeRepo:
    """Stands in for ResumeRepo: BM25 returns the chunks in relevance order."""

    def __init__(self, session=None):
        self._chunks = [
            _Chunk(uuid.uuid4(), DUP_A, position=0),
            _Chunk(uuid.uuid4(), DUP_B, position=1),
            _Chunk(uuid.uuid4(), DUP_C, position=2),
            _Chunk(uuid.uuid4(), DISTINCT_1, position=3),
            _Chunk(uuid.uuid4(), DISTINCT_2, position=4),
        ]

    async def search_chunks_fts(self, resume_id, query, limit=20):
        return list(self._chunks)[:limit]

    async def fetch_chunks(self, resume_id):
        return list(self._chunks)


def _patch_retriever(monkeypatch, *, use_mmr: bool, top_k: int = 3):
    """Fake out every I/O edge of `retrieve` (embedder, vector store, repo,
    cross-encoder) so the test exercises the ranking pipeline only."""
    async def _no_vector_hits(*a, **kw):
        return []

    monkeypatch.setattr(retriever.embedder, "embed_one", lambda text: [0.1] * 8)
    monkeypatch.setattr(retriever.store, "query", _no_vector_hits)
    monkeypatch.setattr(retriever, "ResumeRepo", _FakeRepo)
    monkeypatch.setattr(retriever, "_reranker", lambda: None)
    monkeypatch.setattr(cfg.reranker, "enabled", False)
    monkeypatch.setattr(cfg.rag, "top_k_rerank", top_k)
    monkeypatch.setattr(cfg.rag, "top_k_retrieve", 20)
    monkeypatch.setattr(cfg.advanced_rag, "use_mmr", use_mmr)


def _run_retrieve():
    async def go():
        return await retriever.retrieve("kafka work", resume_id="r1", session=None)

    return asyncio.run(go())


def test_retriever_flag_off_returns_the_redundant_top_k(monkeypatch):
    """Today's behaviour, preserved verbatim when the flag is off."""
    _patch_retriever(monkeypatch, use_mmr=False)
    texts = [h.text for h in _run_retrieve()]
    assert texts == [DUP_A, DUP_B, DUP_C]   # three paraphrases — the old bug


def test_retriever_flag_on_applies_mmr(monkeypatch):
    """The flag is now honoured end-to-end: same query, diverse evidence."""
    _patch_retriever(monkeypatch, use_mmr=True)
    texts = [h.text for h in _run_retrieve()]
    assert len(texts) == 3
    assert texts[0] == DUP_A                                  # relevance kept
    assert sum(t in (DUP_A, DUP_B, DUP_C) for t in texts) == 1
    assert DISTINCT_1 in texts and DISTINCT_2 in texts


def test_retriever_fails_open_when_mmr_raises(monkeypatch):
    """A bug inside MMR must never take retrieval down — it degrades to the
    exact pre-MMR result."""
    _patch_retriever(monkeypatch, use_mmr=True)

    def _boom(*a, **kw):
        raise RuntimeError("mmr exploded")

    monkeypatch.setattr(retriever, "mmr_filter", _boom)
    texts = [h.text for h in _run_retrieve()]
    assert texts == [DUP_A, DUP_B, DUP_C]   # identical to the flag-off result


def test_retriever_still_returns_hits_when_mmr_returns_nothing(monkeypatch):
    _patch_retriever(monkeypatch, use_mmr=True)
    monkeypatch.setattr(retriever, "mmr_filter", lambda *a, **kw: [])
    assert len(_run_retrieve()) == 3


def test_retriever_empty_query_short_circuits(monkeypatch):
    _patch_retriever(monkeypatch, use_mmr=True)

    async def go():
        return await retriever.retrieve("   ", resume_id="r1", session=None)

    assert asyncio.run(go()) == []


# ---- chat-document wiring (app/rag/documents.py) ------------------------
class _FakeStore:
    """Minimal VectorStore: no `hybrid_query`, so `documents` takes the dense
    path. Returns the near-duplicate set in relevance order."""

    async def ensure_collection(self, *a, **kw):
        return None

    async def query(self, collection, *, vector, k=10, filter=None):
        from storage.vectors.base import Hit

        rows = [
            (DUP_A, 0.95), (DUP_B, 0.94), (DUP_C, 0.93),
            (DISTINCT_1, 0.60), (DISTINCT_2, 0.55),
        ]
        return [
            Hit(id=str(i), score=s, payload={"content": c, "filename": "cv.pdf"},
                document=c)
            for i, (c, s) in enumerate(rows)
        ][:k]


def _patch_documents(monkeypatch, *, use_mmr: bool):
    import storage.vectors as _vectors
    from app.rag import query_expand, rerank

    async def _hyde(q):
        return q

    def _fake_rerank(query, hits, top_k):
        # Keep the incoming (score-ordered) order — we're testing MMR, not the
        # cross-encoder, and loading a real one would pull a ~1GB model.
        return [dict(h, rerank_score=h["score"]) for h in hits][:top_k]

    monkeypatch.setattr(documents.embedder, "embed_one", lambda t: [0.1] * 8)
    monkeypatch.setattr(query_expand, "hyde_text", _hyde)
    monkeypatch.setattr(_vectors, "get_vector_store", lambda: _FakeStore())
    monkeypatch.setattr(rerank, "rerank_hits", _fake_rerank)
    monkeypatch.setattr(cfg.advanced_rag, "use_mmr", use_mmr)


def test_chat_retrieval_flag_off_keeps_the_redundant_top_k(monkeypatch):
    _patch_documents(monkeypatch, use_mmr=False)
    hits = asyncio.run(documents.retrieve_chat_hits("conv1", "kafka", k=3))
    assert [h["content"] for h in hits] == [DUP_A, DUP_B, DUP_C]


def test_chat_retrieval_applies_mmr(monkeypatch):
    _patch_documents(monkeypatch, use_mmr=True)
    hits = asyncio.run(documents.retrieve_chat_hits("conv1", "kafka", k=3))
    texts = [h["content"] for h in hits]
    assert texts[0] == DUP_A
    assert sum(t in (DUP_A, DUP_B, DUP_C) for t in texts) == 1
    assert DISTINCT_1 in texts


def test_chat_retrieval_fails_open_when_mmr_raises(monkeypatch):
    _patch_documents(monkeypatch, use_mmr=True)

    from app.rag import mmr as _mmr

    def _boom(*a, **kw):
        raise RuntimeError("mmr exploded")

    monkeypatch.setattr(_mmr, "mmr_filter", _boom)
    hits = asyncio.run(documents.retrieve_chat_hits("conv1", "kafka", k=3))
    assert [h["content"] for h in hits] == [DUP_A, DUP_B, DUP_C]


# ---- select_relevant_chunks: MMR reusing the embeddings it already made --
@dataclass
class _TextChunk:
    text: str


def _patch_select(monkeypatch, *, use_mmr: bool):
    """The near-duplicates here are duplicates in EMBEDDING space, which is what
    this path can afford to measure: it embeds every chunk anyway, so MMR reuses
    those vectors and adds no model call."""
    vectors = {
        DUP_A: [1.0, 0.0, 0.0],
        DUP_B: [0.99, 0.01, 0.0],
        DISTINCT_1: [0.0, 1.0, 0.0],
        DISTINCT_2: [0.0, 0.0, 1.0],
    }
    chunks = [_TextChunk(DUP_A), _TextChunk(DUP_B),
              _TextChunk(DISTINCT_1), _TextChunk(DISTINCT_2)]

    monkeypatch.setattr(documents, "chunk_resume", lambda *a, **kw: chunks)
    monkeypatch.setattr(documents.embedder, "embed",
                        lambda texts: [vectors[t] for t in texts])
    # The query sits closest to the duplicate pair, so relevance alone returns
    # both halves of it.
    monkeypatch.setattr(documents.embedder, "embed_one",
                        lambda t: [1.0, 0.0, 0.0])
    monkeypatch.setattr(cfg.advanced_rag, "use_mmr", use_mmr)


def test_select_relevant_chunks_flag_off_returns_both_duplicates(monkeypatch):
    _patch_select(monkeypatch, use_mmr=False)
    out = asyncio.run(documents.select_relevant_chunks("doc", "q", max_chunks=2))
    assert DUP_A in out and DUP_B in out          # the same evidence twice


def test_select_relevant_chunks_mmr_reuses_vectors_and_diversifies(monkeypatch):
    _patch_select(monkeypatch, use_mmr=True)
    out = asyncio.run(documents.select_relevant_chunks("doc", "q", max_chunks=2))
    assert DUP_A in out
    assert DUP_B not in out                        # near-duplicate dropped
    assert DISTINCT_1 in out                       # replaced by new evidence
