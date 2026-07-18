"""BM25 keyword index — half of the hybrid retrieval pair.

TODO: backed by `rank_bm25` in the venv. Stub until first ingest.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BmHit:
    chunk_id: str
    text: str
    score: float


class Bm25Index:
    def __init__(self) -> None:
        self._docs: list[tuple[str, str]] = []   # (id, text)
        self._bm = None                          # lazily built

    def add(self, doc_id: str, text: str) -> None:
        self._docs.append((doc_id, text))
        self._bm = None                          # mark stale

    def query(self, q: str, top_k: int = 30) -> list[BmHit]:
        # TODO: from rank_bm25 import BM25Okapi
        # if self._bm is None: self._bm = BM25Okapi(tokens for _, tokens in self._docs)
        return []
