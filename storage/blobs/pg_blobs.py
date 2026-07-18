"""Postgres blob store — bytes live in the `blobs` table (bytea).

This is the default for the bundled single-installer deployment: every upload
(chat images, resumes, solve screenshots) and every generated artifact
(documents, rendered images) is stored IN Postgres, so nothing depends on an
external filesystem/volume and everything reloads instantly from the DB.

Keys are the same caller-supplied relative paths the filesystem store used
(`chat_images/<uuid>_name.png`, `documents/<uuid>.pdf`), so call sites and the
references already persisted on Message rows don't change.
"""
from __future__ import annotations

from sqlalchemy import text

from storage.db import get_session_factory


class PostgresBlobs:
    """BlobStore backed by the `blobs` table. Uses the shared async engine.

    An optional `fallback` store (filesystem) is consulted on read-miss so blobs
    written before the Postgres switch still load. New writes go to Postgres.
    """

    def __init__(self, *, fallback=None) -> None:
        self._fallback = fallback

    def _sf(self):
        sf = get_session_factory()
        if sf is None:
            raise RuntimeError("Postgres not ready — blob store unavailable")
        return sf

    async def put(self, path: str, data: bytes, *, filename: str | None = None,
                  content_type: str | None = None, kind: str | None = None) -> str:
        sf = self._sf()
        async with sf() as s:
            await s.execute(
                text(
                    "INSERT INTO blobs (path, data, size, filename, content_type, kind) "
                    "VALUES (:p, cast(:d AS bytea), :n, :fn, :ct, :k) "
                    "ON CONFLICT (path) DO UPDATE SET "
                    "data = EXCLUDED.data, size = EXCLUDED.size, "
                    "filename = COALESCE(EXCLUDED.filename, blobs.filename), "
                    "content_type = COALESCE(EXCLUDED.content_type, blobs.content_type), "
                    "kind = COALESCE(EXCLUDED.kind, blobs.kind), created_at = now()"
                ),
                {"p": path, "d": data, "n": len(data),
                 "fn": filename, "ct": content_type, "k": kind},
            )
            await s.commit()
        return path

    async def get(self, path: str) -> bytes:
        sf = self._sf()
        async with sf() as s:
            row = (await s.execute(
                text("SELECT data FROM blobs WHERE path = :p"), {"p": path}
            )).first()
        if row is None:
            if self._fallback is not None:
                return await self._fallback.get(path)   # legacy on-disk blob
            raise FileNotFoundError(path)
        return bytes(row[0])

    async def get_prefix(self, path: str, length: int) -> bytes:
        if length <= 0:
            return b""
        sf = self._sf()
        async with sf() as s:
            # substring on bytea reads only the prefix server-side, so the full
            # blob (up to 100 MB) is never materialised in this process.
            row = (await s.execute(
                text("SELECT substring(data FROM 1 FOR :n) FROM blobs WHERE path = :p"),
                {"p": path, "n": length},
            )).first()
        if row is None:
            if self._fallback is not None:
                return await self._fallback.get_prefix(path, length)
            raise FileNotFoundError(path)
        return bytes(row[0])

    async def exists(self, path: str) -> bool:
        sf = self._sf()
        async with sf() as s:
            row = (await s.execute(
                text("SELECT 1 FROM blobs WHERE path = :p"), {"p": path}
            )).first()
        if row is None and self._fallback is not None:
            return await self._fallback.exists(path)
        return row is not None

    async def delete(self, path: str) -> None:
        sf = self._sf()
        async with sf() as s:
            await s.execute(text("DELETE FROM blobs WHERE path = :p"), {"p": path})
            await s.commit()

    async def close(self) -> None:
        return None
