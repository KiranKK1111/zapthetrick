"""conversation_search — retrieve from earlier in the thread (vNext §8.4, D).

Once a long thread compacts (§8.4 auto-compaction), the verbatim early turns
leave the window. `conversation_search` lets the loop reach back into them: a
HYBRID search over past turns + compaction digests — lexical BM25 (pure,
deterministic) fused with dense semantic similarity (an INJECTED embedder) via
Reciprocal Rank Fusion, optionally reranked (INJECTED cross-encoder), each result
returned CITED (turn id + source + the exact quote).

The BM25 lane is the always-available floor — it works with no model at all, so
the tool is useful on the dev box and fail-soft on the pod when the embedder is
cold. The dense + rerank lanes are injected seams (the real embedder/reranker run
on-pod). Pure + fail-open. Flag-gated (`tool_loop.conversation_search`, OFF).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+")
_RRF_K = 60          # standard Reciprocal Rank Fusion constant


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.tool_loop, "conversation_search", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class SearchItem:
    """A searchable unit — a past turn or a compaction digest line."""
    id: str
    text: str
    source: str = "turn"          # "turn" | "digest"


@dataclass
class SearchHit:
    id: str
    source: str
    quote: str                    # the cited snippet
    score: float
    rank: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "quote": self.quote,
                "score": round(self.score, 4), "rank": self.rank}


def _tok(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


# --------------------------------------------------------------------------- #
# BM25 (pure, deterministic — the always-available lexical floor)
# --------------------------------------------------------------------------- #
def _bm25_scores(query: str, docs: "list[list[str]]", *,
                 k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_terms = set(_tok(query))
    n = len(docs)
    if not q_terms or n == 0:
        return [0.0] * n
    avgdl = sum(len(d) for d in docs) / n or 1.0
    # Document frequency per query term.
    df: dict[str, int] = {}
    for term in q_terms:
        df[term] = sum(1 for d in docs if term in d)
    scores = [0.0] * n
    for i, d in enumerate(docs):
        if not d:
            continue
        dl = len(d)
        tf: dict[str, int] = {}
        for w in d:
            if w in q_terms:
                tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for term, f in tf.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def _rank_order(scores: "list[float]") -> list[int]:
    """Indices sorted by score desc; ties broken by original order (stable)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _cosine(a, b) -> float:
    try:
        num = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return num / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _best_quote(query: str, text: str, *, max_len: int = 240) -> str:
    """The most query-relevant sentence-ish span of `text` (deterministic —
    picks the segment with the most query-term overlap). Cited verbatim."""
    q = set(_tok(query))
    segs = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    best, best_hits = (text or "").strip(), -1
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        hits = sum(1 for w in set(_tok(seg)) if w in q)
        if hits > best_hits:
            best_hits, best = hits, seg
    return best[:max_len].strip()


def search(query: str, items, *, embed_fn=None, rerank_fn=None,
           top_k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
    """Hybrid dense+BM25 search over `items` (SearchItem or duck-typed
    {id,text,source}) with RRF fusion + optional rerank, returning CITED hits.

    `embed_fn(list[str]) -> list[vec]` and `rerank_fn(query, list[str]) ->
    list[float]` are INJECTED (real models on-pod); both optional — BM25 alone is
    the deterministic floor. Never raises → [] on error."""
    try:
        norm = []
        for it in items or ():
            iid = getattr(it, "id", None) or (it.get("id") if isinstance(it, dict) else None)
            text = getattr(it, "text", None) or (it.get("text") if isinstance(it, dict) else None)
            src = getattr(it, "source", None) or (it.get("source") if isinstance(it, dict) else "turn")
            if iid is None or not (text or "").strip():
                continue
            norm.append((str(iid), text, src or "turn"))
        if not (query or "").strip() or not norm:
            return []

        docs = [_tok(t) for _, t, _ in norm]
        bm25 = _bm25_scores(query, docs)
        bm25_order = _rank_order(bm25)
        # RRF starts from the lexical ranking.
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_order):
            if bm25[idx] > 0:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # Dense lane (injected) — fuse its ranking in when available.
        if embed_fn is not None:
            try:
                vecs = embed_fn([t for _, t, _ in norm])
                qv = embed_fn([query])[0]
                dense = [_cosine(qv, v) for v in vecs]
                for rank, idx in enumerate(_rank_order(dense)):
                    if dense[idx] > 0:
                        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
            except Exception:  # noqa: BLE001 — dense optional, BM25 still stands
                pass

        if not rrf:
            return []
        fused = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))
        cand = [idx for idx, _ in fused[:max(top_k * 3, top_k)]]

        # Rerank lane (injected cross-encoder) over the fused candidates.
        if rerank_fn is not None and cand:
            try:
                rr = rerank_fn(query, [norm[i][1] for i in cand])
                cand = [cand[j] for j in _rank_order(list(rr))]
            except Exception:  # noqa: BLE001
                pass

        hits: list[SearchHit] = []
        for pos, idx in enumerate(cand[:top_k]):
            score = rrf.get(idx, 0.0)
            if score < min_score:
                continue
            iid, text, src = norm[idx]
            hits.append(SearchHit(id=iid, source=src,
                                  quote=_best_quote(query, text),
                                  score=score, rank=pos + 1))
        return hits
    except Exception:  # noqa: BLE001
        return []


def format_citations(hits: "list[SearchHit]") -> str:
    """Render hits as a cited block for the tool result the loop reads."""
    try:
        if not hits:
            return "No earlier matches."
        return "\n".join(
            f"[{h.rank}] ({h.source} {h.id}) {h.quote}" for h in hits)
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["enabled", "SearchItem", "SearchHit", "search", "format_citations"]
