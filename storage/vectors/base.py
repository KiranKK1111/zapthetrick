"""Backend-agnostic vector store interface.

Every method takes a `collection` so one store instance can host many
logical indexes (`resume_chunks_{id}`, `episodic_memory_{user}`,
`semantic_memory_{user}`). Each collection records the
`embedding_model` it was built with so a model swap triggers a re-index
rather than silent corruption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Hit:
    """One vector-search result."""
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    document: str = ""                 # convenience — payload["content"] if present


class VectorStore(Protocol):
    async def ensure_collection(
        self,
        collection: str,
        *,
        vector_size: int,
        embedding_model: str,
        distance: str = "cosine",
    ) -> None: ...

    async def has_collection(self, collection: str) -> bool:
        """True if the collection exists. Used by the startup auto-reindex
        job to spot resumes whose vector index has been lost (Qdrant
        volume wiped, snapshot rolled back, etc.) and rebuild them from
        the durable Postgres chunks."""
        ...

    async def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None: ...

    async def query(
        self,
        collection: str,
        *,
        vector: list[float],
        k: int = 10,
        filter: dict | None = None,
    ) -> list[Hit]: ...

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> None: ...

    async def reset(self, collection: str) -> None: ...
