"""Build the [BlobStore] from `cfg.database.storage.blobs_backend`."""
from __future__ import annotations

from app.core.config_loader import cfg

from .base import BlobStore
from .fs_blobs import FilesystemBlobs
from .minio_blobs import MinioBlobs
from .pg_blobs import PostgresBlobs


_singleton: BlobStore | None = None


def get_blobs() -> BlobStore:
    global _singleton
    if _singleton is not None:
        return _singleton

    s = cfg.database.storage
    # Default to Postgres: uploads + generated artifacts live in the DB so the
    # bundled app needs no external blob volume and everything reloads instantly.
    backend = (s.blobs_backend or "postgres").lower()
    if backend == "minio":
        _singleton = MinioBlobs(
            endpoint=s.minio_endpoint or "localhost:9000",
            access_key=s.minio_access or "",
            secret_key=s.minio_secret or "",
        )
    elif backend in ("filesystem", "fs", "file"):
        _singleton = FilesystemBlobs(root=s.blobs_path)
    else:  # "postgres" / "pg" (default)
        # Read-fallback to the filesystem so blobs written before the switch
        # still load; new writes go to Postgres.
        _singleton = PostgresBlobs(fallback=FilesystemBlobs(root=s.blobs_path))
    return _singleton


async def close_blobs() -> None:
    global _singleton
    if _singleton is not None:
        await _singleton.close()
    _singleton = None
