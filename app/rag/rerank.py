"""Cross-encoder reranking.

A bi-encoder (the embedder) is fast but coarse; a cross-encoder scores the
(query, chunk) pair jointly and is far more precise. We retrieve a wide set
with hybrid search, then rerank with `BAAI/bge-reranker-base` and keep the best.

The model is heavy (~1 GB) and CPU-bound, so it's loaded once (lru_cache) and
callers should run `rerank_hits` via `asyncio.to_thread`.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config_loader import cfg

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    # local-first: cached model skips the online HEAD check; un-cached → fetch.
    try:
        return CrossEncoder(cfg.reranker.model, device=cfg.embeddings.device,
                            local_files_only=True)
    except Exception:  # noqa: BLE001
        return CrossEncoder(cfg.reranker.model, device=cfg.embeddings.device)


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Reorder `hits` (each a dict with a 'content' key) by cross-encoder
    relevance to `query`; return the top_k. Falls back to the input order on
    any failure (model missing, etc.) so retrieval never hard-fails."""
    if not hits:
        return []
    if not getattr(cfg.reranker, "enabled", True) or len(hits) <= 1:
        return hits[:top_k]
    try:
        pairs = [(query, (h.get("content") or "")) for h in hits]
        scores = _cross_encoder().predict(pairs)
        order = sorted(
            range(len(hits)), key=lambda i: float(scores[i]), reverse=True
        )
        ranked = []
        for i in order[:top_k]:
            h = dict(hits[i])
            h["rerank_score"] = float(scores[i])
            ranked.append(h)
        return ranked
    except Exception as exc:  # noqa: BLE001 — reranking is best-effort
        log.info("rerank failed, keeping fused order: %s", exc)
        return hits[:top_k]


__all__ = ["rerank_hits"]
