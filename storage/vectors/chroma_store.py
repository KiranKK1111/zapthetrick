"""Chroma adapter — alternative vector store.

Kept around because (a) the existing Phase-1 install uses it, and
(b) Architecture2.md envisions an embedded mode for single-binary
shipping. Same shape as [PgVectorStore] so the route layer doesn't care.

Chroma's Python client is synchronous; we run hot calls in a worker
thread via `asyncio.to_thread`. For Architecture2.md's embedded-DB
direction (LanceDB / sqlite-vec) we'd add another adapter alongside.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

# Kill Chroma's anonymous PostHog telemetry BEFORE `import chromadb`
# runs anywhere in the process. Belt + suspenders:
#
#   1. Env vars — disable via Settings hint and tell Chroma to use the
#      no-op telemetry client (when available).
#   2. Monkey-patch — Chroma 0.6.3's `Posthog.capture` skips the
#      anonymized-telemetry check entirely and calls posthog directly
#      with 3 positional args. The new posthog SDK rejects that
#      signature, the exception bubbles up to `_direct_capture`, and
#      we get the "Failed to send telemetry event ..." spam. The only
#      reliable fix is to replace the broken `capture` with a no-op.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault(
    "CHROMA_TELEMETRY_IMPL",
    "chromadb.telemetry.product.NoopProductTelemetryClient",
)


def _silence_chroma_telemetry() -> None:
    """Monkey-patch `Posthog._direct_capture` to a no-op.

    Called from `_get_client` AFTER `import chromadb` — running it at
    module-import time would force chromadb to load even when the
    Chroma backend isn't selected.
    """
    try:
        from chromadb.telemetry.product import posthog as _ph

        def _noop(self, event) -> None:
            return None

        _ph.Posthog._direct_capture = _noop  # type: ignore[assignment]
        # Also silence the logger that prints the original error in case
        # any other code path slips a capture through.
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(
            logging.CRITICAL
        )
    except Exception:
        # Older / future Chroma might restructure this module. Falling
        # back to the env vars is fine.
        pass


from .base import Hit


log = logging.getLogger(__name__)


_DISTANCE_TO_CHROMA = {
    "cosine": "cosine",
    "dot": "ip",
    "euclid": "l2",
}


class ChromaStore:
    def __init__(self, *, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        self._client = None
        self._lock = asyncio.Lock()
        self._collections: dict[str, object] = {}

    async def _get_client(self):
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    import chromadb
                    from chromadb.config import Settings as _ChromaSettings
                except ImportError as exc:
                    raise RuntimeError(
                        "chromadb is not installed. Run: pip install chromadb"
                    ) from exc

                # Patch chromadb's broken telemetry before instantiating
                # the client. `_get_client` is the *only* place we
                # construct a chromadb anything, so a single patch
                # here kills the noise for startup + create + query.
                _silence_chroma_telemetry()

                def _build():
                    Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                    return chromadb.PersistentClient(
                        path=str(Path(self.persist_dir).resolve()),
                        settings=_ChromaSettings(anonymized_telemetry=False),
                    )

                self._client = await asyncio.to_thread(_build)
        return self._client

    async def has_collection(self, collection: str) -> bool:
        if collection in self._collections:
            return True
        try:
            client = await self._get_client()

            def _check() -> bool:
                # Chroma raises ValueError when the collection is missing;
                # any non-error return means it exists.
                try:
                    client.get_collection(name=collection)
                    return True
                except Exception:
                    return False

            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def ensure_collection(
        self,
        collection: str,
        *,
        vector_size: int,
        embedding_model: str,
        distance: str = "cosine",
    ) -> None:
        if collection in self._collections:
            return
        client = await self._get_client()
        space = _DISTANCE_TO_CHROMA.get(distance.lower(), "cosine")

        def _create():
            return client.get_or_create_collection(
                name=collection,
                metadata={
                    "hnsw:space": space,
                    "embedding_model": embedding_model,
                    "vector_size": vector_size,
                },
            )

        self._collections[collection] = await asyncio.to_thread(_create)

    async def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        if not ids:
            return
        coll = self._collections.get(collection)
        if coll is None:
            raise RuntimeError(
                f"chroma collection {collection!r} not ensured before upsert"
            )
        docs = [p.get("content", "") for p in payloads]
        await asyncio.to_thread(
            coll.upsert,
            ids=ids,
            documents=docs,
            embeddings=vectors,
            metadatas=payloads,
        )

    async def query(
        self,
        collection: str,
        *,
        vector: list[float],
        k: int = 10,
        filter: dict | None = None,
    ) -> list[Hit]:
        coll = self._collections.get(collection)
        if coll is None:
            return []

        def _q():
            return coll.query(
                query_embeddings=[vector],
                n_results=k,
                where=filter or None,
            )

        res = await asyncio.to_thread(_q)
        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[Hit] = []
        for i, hid in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            # Chroma returns distance; convert to similarity for callers
            # that treat "higher is better".
            d = dists[i] if i < len(dists) else 0.0
            sim = 1.0 - float(d) if d is not None else 0.0
            out.append(Hit(id=hid, score=sim, payload=meta or {}, document=doc))
        return out

    async def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> None:
        coll = self._collections.get(collection)
        if coll is None:
            return
        await asyncio.to_thread(coll.delete, ids=ids, where=filter)

    async def reset(self, collection: str) -> None:
        client = await self._get_client()
        try:
            await asyncio.to_thread(client.delete_collection, name=collection)
        except Exception as exc:
            log.warning("chroma reset(%s) ignored: %s", collection, exc)
        self._collections.pop(collection, None)

    async def close(self) -> None:
        self._client = None
        self._collections.clear()
