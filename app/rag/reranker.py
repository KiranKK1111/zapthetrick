"""Cross-encoder reranker — BAAI/bge-reranker-base by default.

TODO: lazy-load via `sentence_transformers.CrossEncoder` and rank with
the joint encoder. Until then, this passes hits through unchanged so the
pipeline composes end-to-end.
"""
from __future__ import annotations


class Reranker:
    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model
        self._model = None

    async def rerank(self, query: str, hits, *, top_k: int = 10):
        # TODO: real cross-encoder pass.
        return list(hits)[:top_k]
