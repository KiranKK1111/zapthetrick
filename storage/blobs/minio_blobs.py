"""MinIO / S3 blob store — alternative.

The MinIO client is sync; we hot-path it via `asyncio.to_thread`.
Endpoint and creds come from `cfg.database.storage`. The bucket is
created on first connect when missing.

TODO: server-side encryption headers per the spec's "encryption at
rest" note — `Minio.put_object` accepts an `sse` param.
"""
from __future__ import annotations

import asyncio
from io import BytesIO


class MinioBlobs:
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "zapthetrick",
        secure: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self._client = None
        self._lock = asyncio.Lock()

    async def _get(self):
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    from minio import Minio
                except ImportError as exc:
                    raise RuntimeError(
                        "minio is not installed. Run: pip install minio"
                    ) from exc

                def _build():
                    c = Minio(
                        self.endpoint,
                        access_key=self.access_key,
                        secret_key=self.secret_key,
                        secure=self.secure,
                    )
                    if not c.bucket_exists(self.bucket):
                        c.make_bucket(self.bucket)
                    return c

                self._client = await asyncio.to_thread(_build)
        return self._client

    async def put(self, path: str, data: bytes) -> str:
        client = await self._get()

        def _put() -> None:
            client.put_object(
                self.bucket,
                path,
                BytesIO(data),
                length=len(data),
            )

        await asyncio.to_thread(_put)
        return f"s3://{self.bucket}/{path}"

    async def get(self, path: str) -> bytes:
        client = await self._get()

        def _get() -> bytes:
            resp = client.get_object(self.bucket, path)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return await asyncio.to_thread(_get)

    async def get_prefix(self, path: str, length: int) -> bytes:
        if length <= 0:
            return b""
        client = await self._get()

        def _get() -> bytes:
            # Range request — S3/MinIO returns only the first `length` bytes.
            resp = client.get_object(self.bucket, path, offset=0, length=length)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return await asyncio.to_thread(_get)

    async def exists(self, path: str) -> bool:
        client = await self._get()

        def _stat() -> bool:
            try:
                client.stat_object(self.bucket, path)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_stat)

    async def delete(self, path: str) -> None:
        client = await self._get()
        await asyncio.to_thread(client.remove_object, self.bucket, path)

    async def close(self) -> None:
        self._client = None
