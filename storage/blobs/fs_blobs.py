"""Filesystem blob store — the default.

Files land under `cfg.database.storage.blobs_path`. Parents created on
demand. Async I/O via `asyncio.to_thread` so blocking disk ops don't
stall the event loop.
"""
from __future__ import annotations

import asyncio
from pathlib import Path


class FilesystemBlobs:
    def __init__(self, *, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        # Strip leading separators so callers can't escape `root`.
        rel = Path(path.lstrip("/\\"))
        full = (self.root / rel).resolve()
        # Defense against `..` traversal — must stay under root.
        if self.root not in full.parents and full != self.root:
            raise PermissionError(f"blob path escapes root: {path!r}")
        return full

    async def put(self, path: str, data: bytes) -> str:
        full = self._resolve(path)

        def _write() -> None:
            full.parent.mkdir(parents=True, exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)

        await asyncio.to_thread(_write)
        return str(full)

    async def get(self, path: str) -> bytes:
        full = self._resolve(path)
        return await asyncio.to_thread(full.read_bytes)

    async def get_prefix(self, path: str, length: int) -> bytes:
        if length <= 0:
            return b""
        full = self._resolve(path)

        def _read() -> bytes:
            with open(full, "rb") as f:
                return f.read(length)

        return await asyncio.to_thread(_read)

    async def exists(self, path: str) -> bool:
        full = self._resolve(path)
        return await asyncio.to_thread(full.exists)

    async def delete(self, path: str) -> None:
        full = self._resolve(path)

        def _unlink() -> None:
            try:
                full.unlink()
            except FileNotFoundError:
                pass

        await asyncio.to_thread(_unlink)

    async def close(self) -> None:
        return None
