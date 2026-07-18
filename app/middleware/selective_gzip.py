"""Selective gzip ASGI middleware (smooth-streaming-rendering R37).

Why not Starlette's stock ``GZipMiddleware``? It wraps *every* response,
including streaming ones, in a streaming gzip compressor. For Server-Sent
Events that is fatal: gzip's internal buffering coalesces many small SSE
``data:`` frames into the compressor window, so the client receives tokens in
large bursts (or only at stream end) — the exact opposite of smooth streaming.

This middleware compresses ONLY complete, non-streaming, compressible responses
(JSON, plain text, HTML, …) and explicitly passes ``text/event-stream`` (and any
already-encoded or non-compressible response) straight through, unbuffered. SSE
latency is therefore unchanged while ordinary JSON API payloads still shrink on
the wire.

Registration (in ``app/main.py``)::

    from app.middleware.selective_gzip import SelectiveGZipMiddleware
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=500)
"""
from __future__ import annotations

import gzip

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Content types worth compressing. ``text/event-stream`` is deliberately absent
# and is also rejected explicitly below so a future "text/*" entry can't let it
# slip through.
_COMPRESSIBLE_PREFIXES = (
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-ndjson",
    "text/",
    "image/svg+xml",
)


def _is_compressible(content_type: str) -> bool:
    ct = content_type.split(";", 1)[0].strip().lower()
    if not ct or ct == "text/event-stream":
        return False
    return ct.startswith(_COMPRESSIBLE_PREFIXES)


def _accepts_gzip(scope: Scope) -> bool:
    for key, value in scope.get("headers", []):
        if key == b"accept-encoding":
            return b"gzip" in value.lower()
    return False


class SelectiveGZipMiddleware:
    """Compress complete compressible responses; stream everything else."""

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = 500,
        compresslevel: int = 6,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _accepts_gzip(scope):
            await self.app(scope, receive, send)
            return
        responder = _GZipResponder(
            self.app, self.minimum_size, self.compresslevel
        )
        await responder(scope, receive, send)


class _GZipResponder:
    def __init__(
        self, app: ASGIApp, minimum_size: int, compresslevel: int
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel
        self._send: Send | None = None
        self._start: Message | None = None
        self._passthrough = False
        self._body = bytearray()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self._send = send
        await self.app(scope, receive, self._send_filtered)

    async def _send_filtered(self, message: Message) -> None:
        assert self._send is not None
        kind = message["type"]

        if kind == "http.response.start":
            self._start = message
            headers = Headers(raw=message.get("headers", []))
            already_encoded = bool(headers.get("content-encoding"))
            compressible = _is_compressible(headers.get("content-type", ""))
            # Stream non-compressible / SSE / pre-encoded responses untouched.
            self._passthrough = already_encoded or not compressible
            if self._passthrough:
                await self._send(message)
            # Otherwise hold the start until the full body is buffered so we can
            # set an accurate Content-Length / Content-Encoding.
            return

        if kind != "http.response.body":
            await self._send(message)
            return

        if self._passthrough:
            await self._send(message)
            return

        self._body.extend(message.get("body", b""))
        if message.get("more_body", False):
            return  # keep buffering this compressible response

        raw = bytes(self._body)
        assert self._start is not None
        if len(raw) < self.minimum_size:
            await self._send(self._start)
            await self._send(
                {"type": "http.response.body", "body": raw, "more_body": False}
            )
            return

        compressed = gzip.compress(raw, compresslevel=self.compresslevel)
        headers = MutableHeaders(raw=self._start["headers"])
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed))
        vary = headers.get("Vary")
        if not vary:
            headers["Vary"] = "Accept-Encoding"
        elif "accept-encoding" not in vary.lower():
            headers["Vary"] = f"{vary}, Accept-Encoding"
        await self._send(self._start)
        await self._send(
            {"type": "http.response.body", "body": compressed, "more_body": False}
        )
