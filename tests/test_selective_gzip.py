"""Tests for SelectiveGZipMiddleware (R37).

Verifies that ordinary JSON/text responses are gzipped while Server-Sent Event
streams pass through uncompressed and unbuffered. These build a tiny standalone
FastAPI app so they never import `app.main` (which loads ML models and hangs).
"""
import gzip

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.testclient import TestClient

from app.middleware.selective_gzip import (
    SelectiveGZipMiddleware,
    _is_compressible,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=50)

    @app.get("/json")
    def _json():
        return {"data": "x" * 500}

    @app.get("/tiny")
    def _tiny():
        return PlainTextResponse("hi")

    @app.get("/sse")
    def _sse():
        def gen():
            for i in range(5):
                yield f"data: chunk-{i}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def test_json_is_gzipped_when_accepted():
    client = TestClient(_app())
    r = client.get("/json", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in r.headers.get("vary", "").lower()
    # TestClient transparently decodes; payload must round-trip.
    assert r.json()["data"] == "x" * 500


def test_not_compressed_without_accept_encoding():
    client = TestClient(_app())
    r = client.get("/json", headers={"accept-encoding": "identity"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers


def test_tiny_response_below_threshold_not_compressed():
    client = TestClient(_app())
    r = client.get("/tiny", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert r.text == "hi"


def test_sse_is_not_compressed():
    client = TestClient(_app())
    r = client.get("/sse", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    # SSE must never be gzip-encoded (would break incremental streaming).
    assert "content-encoding" not in r.headers
    assert "chunk-0" in r.text and "chunk-4" in r.text


def test_is_compressible_predicate():
    assert _is_compressible("application/json")
    assert _is_compressible("text/plain; charset=utf-8")
    assert _is_compressible("image/svg+xml")
    # SSE and binary types must be rejected.
    assert not _is_compressible("text/event-stream")
    assert not _is_compressible("application/octet-stream")
    assert not _is_compressible("image/png")
    assert not _is_compressible("")


def test_gzip_payload_is_actually_smaller():
    # Sanity: a real gzip of the body is meaningfully smaller than the raw.
    body = (b"x" * 500)
    assert len(gzip.compress(body)) < len(body)
