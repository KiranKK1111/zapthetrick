"""MMR diversity filter — promotes varied evidence, drops near-duplicates.

Maximal Marginal Relevance: greedily pick the next hit that maximizes

    λ·rel(query, hit) − (1−λ)·max sim(hit, already_picked)

so the k we return covers several distinct pieces of evidence instead of
k paraphrases of the same one.

WHERE THE TWO TERMS COME FROM (both are free — MMR adds no model call):

  * `rel` — the score the hit ALREADY carries (cross-encoder rerank score,
    or the RRF score when the reranker is off). Min-max normalised to
    [0, 1] so λ trades off against a comparable similarity.

  * `sim` — redundancy between two candidates. Two backends:

      1. **cosine** when the caller can hand us vectors for free
         (`vectors=`, or hits carrying `.embedding` / `.vector` /
         `["embedding"]`). `documents.select_relevant_chunks` embeds every
         chunk anyway, so it gets true vector MMR at zero extra cost.

      2. **token cosine** (bag-of-words) when it can't. The hot retrieval
         path is in this bucket ON PURPOSE: the vector store never returns
         stored vectors (`storage.vectors.base.Hit` has no vector field)
         and `resume_chunks` has no embedding column, so the only way to
         get vectors there is to re-embed the candidate pool — with
         bge-m3 on CPU (the configured embedder) that is hundreds of ms
         added to every query, a latency regression that MMR does not pay
         for. Token cosine costs microseconds on a ≤20-chunk pool and
         catches exactly the near-duplicates this pipeline actually
         produces: overlapping chunk windows (`rag.chunk_overlap`) and
         parent/child hierarchical chunks, which are lexically
         near-identical by construction.

`embed_fn=` is available for callers that genuinely want to pay for
embeddings, but nothing on the hot path passes it.

Every entry point is total: bad input degrades to "top-k by score", never
raises. Callers still wrap it (fail-open) so a bug here can't take
retrieval down.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any


DEFAULT_LAMBDA = 0.7

# A candidate this similar to something already picked is a NEAR-DUPLICATE and
# is dropped outright, rather than competing on relevance.
#
# λ alone does not dedupe: a paraphrase of the top hit is also highly relevant,
# so at λ=0.7 it still outscores a genuinely-new-but-less-relevant chunk and the
# top-k stays redundant. The cutoff is what makes "drops near-duplicates" true.
# 0.9 is deliberately high: unrelated chunks sit near 0.1-0.3 on token cosine,
# and dense embedders (bge-m3) put merely-RELATED text as high as 0.7, so only
# genuine restatements clear the bar. λ=1.0 (pure relevance) disables it.
DEFAULT_SIM_THRESHOLD = 0.9

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---- similarity ---------------------------------------------------------
def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two vectors, clamped to [0, 1] (negatives mean
    'not redundant' for our purposes)."""
    n = min(len(u), len(v))
    if n == 0:
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for i in range(n):
        a = float(u[i])
        b = float(v[i])
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (math.sqrt(nu) * math.sqrt(nv))))


def _tokens(text: str) -> Counter:
    return Counter(_TOKEN_RE.findall((text or "").lower()))


def token_cosine(a: str, b: str) -> float:
    """Bag-of-words cosine between two texts, in [0, 1].

    1.0 for identical text, ~0.8+ for an overlapping chunk window, ~0.1-0.3
    for two genuinely different chunks of the same resume.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    common = ta.keys() & tb.keys()
    if not common:
        return 0.0
    dot = sum(ta[t] * tb[t] for t in common)
    na = math.sqrt(sum(c * c for c in ta.values()))
    nb = math.sqrt(sum(c * c for c in tb.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ---- core selection -----------------------------------------------------
def mmr_select(
    relevance: Sequence[float],
    *,
    vectors: Sequence[Sequence[float]] | None = None,
    texts: Sequence[str] | None = None,
    top_k: int = 5,
    lambda_: float = DEFAULT_LAMBDA,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> list[int]:
    """Greedy MMR over candidate INDICES. Returns the picked indices, in the
    order MMR picked them (most relevant first).

    `vectors` (cosine) is used when supplied; otherwise `texts` (token
    cosine). With neither, this degenerates to top-k by relevance.

    Candidates that are ≥ `sim_threshold` similar to an already-picked hit are
    dropped, so the result can be SHORTER than `top_k` — that is the point: five
    slots of the same passage is worse context than three distinct ones.
    """
    n = len(relevance)
    if n == 0 or top_k <= 0:
        return []
    if n <= 1:
        return [0]

    use_vectors = bool(vectors) and len(vectors) == n
    use_texts = (not use_vectors) and bool(texts) and len(texts) == n

    rel = _normalise(relevance)
    lam = min(1.0, max(0.0, float(lambda_)))

    # Similarity is O(k·n) lookups on a small pool — memoise the pairs we touch.
    sim_cache: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        hit = sim_cache.get(key)
        if hit is None:
            if use_vectors:
                hit = cosine(vectors[key[0]], vectors[key[1]])
            elif use_texts:
                hit = token_cosine(texts[key[0]], texts[key[1]])
            else:
                hit = 0.0
            sim_cache[key] = hit
        return hit

    # λ=1 is "pure relevance" — callers asking for that get no dedupe at all.
    cutoff = math.inf if lam >= 1.0 else min(1.0, max(0.0, float(sim_threshold)))

    picked: list[int] = []
    remaining = list(range(n))
    while remaining and len(picked) < top_k:
        best = None
        best_score = -math.inf
        redundant: list[int] = []
        for i in remaining:
            redundancy = max((sim(i, j) for j in picked), default=0.0)
            if picked and redundancy >= cutoff:
                redundant.append(i)     # a restatement of something we kept
                continue
            score = lam * rel[i] - (1.0 - lam) * redundancy
            # Strict `>` keeps the ORIGINAL rank order as the tie-breaker, so
            # the first pick is always the top-ranked hit.
            if score > best_score:
                best_score = score
                best = i
        for i in redundant:
            remaining.remove(i)
        if best is None:            # everything left was a near-duplicate
            break
        picked.append(best)
        remaining.remove(best)
    return picked


def _normalise(scores: Sequence[float]) -> list[float]:
    """Min-max the relevance scores into [0, 1].

    Cross-encoder logits are unbounded (roughly -11..+11 for bge-reranker)
    while RRF scores sit around 0.016 — either would swamp or vanish against a
    [0, 1] similarity if we used them raw, making λ meaningless.
    """
    vals = [float(s) for s in scores]
    lo, hi = min(vals), max(vals)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= 1e-12:
        # All-equal (or degenerate) scores: relevance carries no signal, so let
        # diversity decide, with rank order as the tie-break.
        return [1.0] * len(vals)
    span = hi - lo
    return [(v - lo) / span for v in vals]


# ---- object-facing API --------------------------------------------------
def mmr_filter(
    query: str,
    hits,
    *,
    top_k: int = 5,
    embedder=None,
    lambda_: float = DEFAULT_LAMBDA,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    vectors: Sequence[Sequence[float]] | None = None,
    embed_fn=None,
):
    """Diversity-rerank `hits` down to `top_k`, dropping near-duplicates.

    `hits` may be objects (`RetrievalHit`: `.text` / `.score`) or dicts
    (`{"content"|"text", "score"|"rerank_score"}`) — both shapes are in use in
    this codebase. Exact-duplicate texts are dropped outright (that was this
    module's original behaviour and it is still the cheapest correct move).

    Vectors are used when they are already available: passed via `vectors=`, or
    carried on the hits themselves. `embed_fn` (a callable `list[str] ->
    list[list[float]]`, or a module with `.embed`) is an OPT-IN escape hatch for
    callers willing to pay for embeddings; `embedder=` is the legacy name for
    the same thing and is accepted for backwards compatibility. Nothing on the
    hot retrieval path passes either — see the module docstring.

    Returns the picked hit objects (the same objects, not copies), in MMR order.
    """
    items = list(hits or [])
    if top_k <= 0:
        return []
    if len(items) <= 1:
        return items[:top_k]

    texts = [_text_of(h) for h in items]

    # Exact duplicates: keep the first (highest-ranked) occurrence only.
    seen: set[str] = set()
    keep: list[int] = []
    for i, t in enumerate(items):
        key = texts[i]
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    if len(keep) <= 1:
        return [items[i] for i in keep][:top_k]

    kept_items = [items[i] for i in keep]
    kept_texts = [texts[i] for i in keep]
    kept_scores = [_score_of(h, rank) for rank, h in enumerate(kept_items)]

    vecs = _vectors_for(
        kept_items,
        kept_texts,
        [vectors[i] for i in keep] if vectors and len(vectors) == len(items) else None,
        embed_fn if embed_fn is not None else embedder,
    )

    picked = mmr_select(
        kept_scores,
        vectors=vecs,
        texts=kept_texts,
        top_k=top_k,
        lambda_=lambda_,
        sim_threshold=sim_threshold,
    )
    return [kept_items[i] for i in picked]


# ---- extraction helpers -------------------------------------------------
def _text_of(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("content") or hit.get("text") or "")
    for attr in ("text", "content", "document"):
        val = getattr(hit, attr, None)
        if isinstance(val, str):
            return val
    return ""


def _score_of(hit: Any, rank: int) -> float:
    """The relevance the hit already carries; falls back to its rank."""
    if isinstance(hit, dict):
        for key in ("rerank_score", "score"):
            val = hit.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    else:
        for attr in ("rerank_score", "score"):
            val = getattr(hit, attr, None)
            if isinstance(val, (int, float)):
                return float(val)
    return 1.0 / (rank + 1)


def _vector_of(hit: Any) -> list[float] | None:
    if isinstance(hit, dict):
        val = hit.get("embedding") or hit.get("vector")
    else:
        val = getattr(hit, "embedding", None) or getattr(hit, "vector", None)
    if isinstance(val, (list, tuple)) and val:
        return [float(x) for x in val]
    return None


def _vectors_for(items, texts, supplied, embed_fn) -> list[list[float]] | None:
    """Vectors for the candidates, or None to fall back to token cosine.

    Order of preference: explicitly supplied → carried on the hits → an opt-in
    `embed_fn`. Never embeds unless the caller handed us a way to.
    """
    if supplied and len(supplied) == len(items):
        return [list(v) for v in supplied]

    carried = [_vector_of(h) for h in items]
    if all(v is not None for v in carried):
        return carried  # type: ignore[return-value]

    if embed_fn is None:
        return None
    fn = getattr(embed_fn, "embed", embed_fn)
    if not callable(fn):
        return None
    try:
        out = fn(list(texts))
    except Exception:  # noqa: BLE001 — opt-in path; degrade to token cosine
        return None
    if not out or len(out) != len(items):
        return None
    return [list(v) for v in out]


__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_SIM_THRESHOLD",
    "cosine",
    "mmr_filter",
    "mmr_select",
    "token_cosine",
]
