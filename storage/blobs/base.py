"""Backend-agnostic blob interface — resumes, screenshots, audio clips.

Callers pass relative paths (`resumes/{id}.pdf`, `screenshots/.../ts.png`)
and the implementation resolves them against the configured root.

Per DataBaseArchitecture.md: blobs never belong inside the database.
Postgres stores the *path* (`Resume.file_path`); the bytes live here.
"""
from __future__ import annotations

from typing import Protocol


class BlobStore(Protocol):
    async def put(self, path: str, data: bytes) -> str:
        """Write `data` under `path`. Returns the canonical reference
        the caller should store in Postgres (`file_path` for FS, a key
        for MinIO)."""
        ...

    async def get(self, path: str) -> bytes: ...

    async def get_prefix(self, path: str, length: int) -> bytes:
        """Read at most the first `length` bytes of `path`. Backends that can
        range-read (Postgres `substring`, file seek, S3 range) avoid pulling the
        whole blob into memory — used for bounded previews of large files."""
        ...

    async def exists(self, path: str) -> bool: ...
    async def delete(self, path: str) -> None: ...
    async def close(self) -> None: ...
