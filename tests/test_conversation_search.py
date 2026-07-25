"""Tests for conversation_search hybrid retrieval (vNext §8.4, Stage 7 D)."""
from __future__ import annotations

import app.memory.conversation_search as CS


def _items():
    return [
        CS.SearchItem("m1", "We decided to use Postgres for durable storage."),
        CS.SearchItem("m2", "The weather was nice and we talked about lunch."),
        CS.SearchItem("d1", "Decisions: chose Postgres database; port 8080",
                      source="digest"),
    ]


# ---- BM25 lexical floor (no models) ---------------------------------------
def test_bm25_ranks_relevant_turn_first():
    hits = CS.search("which postgres database storage", _items(), top_k=2)
    assert hits
    assert hits[0].id in ("m1", "d1")     # a postgres item wins, not the weather
    assert hits[0].rank == 1


def test_hits_are_cited_with_source_and_quote():
    hits = CS.search("postgres storage", _items(), top_k=1)
    h = hits[0]
    assert h.source in ("turn", "digest")
    assert "Postgres" in h.quote          # the quote is the relevant span
    d = h.to_dict()
    assert set(d) == {"id", "source", "quote", "score", "rank"}


def test_digest_items_are_searchable():
    hits = CS.search("port 8080", _items(), top_k=3)
    assert any(h.id == "d1" and h.source == "digest" for h in hits)


def test_empty_query_or_items_returns_empty():
    assert CS.search("", _items()) == []
    assert CS.search("postgres", []) == []


def test_no_lexical_match_returns_empty():
    hits = CS.search("zzzznonexistentterm", _items())
    assert hits == []


# ---- dense lane (injected) ------------------------------------------------
def test_dense_lane_fuses_semantic_match():
    # An embedder that only "understands" postgres → boosts m1/d1 semantically
    # even when the query wording differs.
    def emb(texts):
        return [[1.0, 0.0] if "postgres" in t.lower() else [0.0, 1.0]
                for t in texts]
    hits = CS.search("postgres", _items(), embed_fn=emb, top_k=3)
    ids = [h.id for h in hits]
    assert "m1" in ids and "d1" in ids


def test_dense_failure_falls_back_to_bm25():
    def boom(texts):
        raise RuntimeError("embedder cold")
    hits = CS.search("postgres storage", _items(), embed_fn=boom, top_k=2)
    assert hits                            # BM25 still produced results


# ---- rerank lane (injected) -----------------------------------------------
def test_rerank_reorders_candidates():
    # A reranker that puts the digest item on top regardless of fusion order.
    def rr(query, texts):
        return [1.0 if "port 8080" in t else 0.0 for t in texts]
    hits = CS.search("postgres database", _items(), rerank_fn=rr, top_k=3)
    assert hits[0].id == "d1"


def test_rerank_failure_is_safe():
    def boom(query, texts):
        raise RuntimeError("reranker down")
    hits = CS.search("postgres", _items(), rerank_fn=boom, top_k=2)
    assert hits                            # fusion order preserved, no crash


# ---- robustness -----------------------------------------------------------
def test_duck_typed_dict_items():
    items = [{"id": "x1", "text": "Postgres was chosen here.", "source": "turn"}]
    hits = CS.search("postgres", items, top_k=1)
    assert hits and hits[0].id == "x1"


def test_min_score_filters():
    hits = CS.search("postgres storage", _items(), top_k=5, min_score=999.0)
    assert hits == []                      # nothing clears an impossible bar


def test_never_raises_on_garbage():
    assert CS.search("q", [{"id": None, "text": None}]) == []


def test_format_citations_empty():
    assert CS.format_citations([]) == "No earlier matches."
