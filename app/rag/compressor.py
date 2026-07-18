"""Contextual compressor — sentence-level relevance scoring.

Drops sentences that don't contribute to the query, keeping ~30–50%
of the original tokens (Architecture.md §3).

TODO: real scoring via cross-encoder on (query, sentence) pairs.
"""
from __future__ import annotations


class ContextualCompressor:
    def __init__(self, target_ratio: float = 0.4) -> None:
        self.target_ratio = target_ratio

    async def compress(self, query: str, hits, *, target_ratio: float | None = None):
        ratio = target_ratio or self.target_ratio
        # TODO: real per-sentence relevance scoring.
        return list(hits)
