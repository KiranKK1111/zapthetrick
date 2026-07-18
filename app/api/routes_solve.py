"""Code-solver endpoints + Solve history.

Streaming:
  POST /api/solve/text         JSON: {problem, language?}              -> SSE
  POST /api/solve/image        multipart: file=<image>, language?       -> SSE

History (mirrors the Chat tab's conversations list):
  GET  /api/solve/sessions             list, newest first
  GET  /api/solve/sessions/{id}        full detail (description + response)
  DELETE /api/solve/sessions/{id}      remove from history

Every Solve click persists one row in `solve_sessions`. The row carries
the user-facing title (auto-derived from the description), the full
problem statement (typed body or OCR'd from the screenshot), the
model's response, and metadata (source, language, latency, models used).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import LLMError
from app.tools import code_solver
from storage import get_session
from storage.blobs import get_blobs
from storage.db import get_session_factory
from storage.repos import SolveRepo
from storage.users import get_default_user_id


router = APIRouter(prefix="/api/solve")


class SolveTextRequest(BaseModel):
    """Body for POST /api/solve/text."""
    problem: str = Field(..., min_length=1)
    language: str | None = None


def _json_default(o):
    """UUID / datetime hook for SSE payloads — without this the first
    frame containing a Postgres-generated UUID raises in `json.dumps`
    and the client sees 'Connection closed while receiving data'."""
    import uuid as _uuid
    from datetime import datetime as _dt

    if isinstance(o, _uuid.UUID):
        return str(o)
    if isinstance(o, _dt):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=_json_default)}\n\n"


def _extract_problem_from_response(response: str, filename: str | None) -> str:
    """Pull a useful problem description out of the streamed answer.

    Used by the single-step image pipeline where the vision model
    reads + solves in one call and we have no separate OCR stage to
    capture. Walks the structured response (Problem / Approach /
    Solution / ...) and returns whatever the model wrote under
    "Problem" — or the response's leading paragraph as a last resort.

    Falls back to a filename-tagged placeholder only when the
    response is empty (model failure).
    """
    text = (response or "").strip()
    if not text:
        return f"(image solve — {filename or 'screenshot'})"

    # The code_solver system prompt asks the model to output sections
    # like `## Problem` / `**Problem**` / `Problem:` followed by the
    # problem text. Try those, in order, before giving up.
    lower = text.lower()
    for marker in ("## problem", "**problem**", "problem:", "## question"):
        idx = lower.find(marker)
        if idx == -1:
            continue
        # Slice from end-of-marker to the next heading (`\n## ` /
        # `\n**` line) or 1500 chars, whichever comes first.
        start = idx + len(marker)
        chunk = text[start : start + 1500]
        # Stop at the next section heading.
        for stop in ("\n## ", "\n**", "\n```", "\nApproach", "\nSolution"):
            stop_idx = chunk.find(stop)
            if stop_idx != -1:
                chunk = chunk[:stop_idx]
                break
        cleaned = chunk.strip(" :\n*")
        if len(cleaned) >= 20:
            return cleaned

    # No structured Problem section — use the leading paragraph.
    para = text.split("\n\n", 1)[0].strip()
    if len(para) >= 20:
        return para[:1500]

    return f"(image solve — {filename or 'screenshot'})"


async def _persist_solve(
    *,
    description: str,
    response: str,
    source: str,
    language: str | None,
    image_path: str | None = None,
    vision_model: str | None = None,
    code_model: str | None = None,
    latency_ms: int,
) -> str | None:
    """Open a fresh session and insert one `solve_sessions` row.

    A fresh session — the route's request-scoped one is gone by the
    time the SSE generator drains. Failures are logged + swallowed so
    a DB hiccup doesn't kill the user's already-rendered answer.
    Returns the new row id (string) on success.
    """
    import logging

    log = logging.getLogger(__name__)
    if not response.strip():
        return None
    factory = get_session_factory()
    if factory is None:
        log.warning("solve persist skipped — SessionFactory is None")
        return None
    try:
        async with factory() as write_session:
            row = await SolveRepo(write_session).create(
                description=description or "(no problem statement captured)",
                response=response,
                user_id=get_default_user_id(),
                language=language,
                source=source,
                image_path=image_path,
                vision_model=vision_model,
                code_model=code_model,
                latency_ms=latency_ms,
            )
            await write_session.commit()
            await write_session.refresh(row)
            solve_id = str(row.id)
            placeholder_title = row.title or ""

        # Architecture: replace the first-line placeholder with a
        # tight LLM-generated title. Fire-and-forget — never blocks
        # the user-visible response and silently keeps the
        # placeholder on failure.
        import asyncio as _asyncio

        from app.solve.auto_title import maybe_title

        _asyncio.create_task(
            maybe_title(
                solve_id,
                description=description or "",
                response=response,
                current_title=placeholder_title,
            ),
            name=f"solve-auto-title-{solve_id}",
        )
        return solve_id
    except Exception as exc:  # noqa: BLE001
        log.warning("solve persist failed: %s", exc)
        return None


@router.post("/text")
async def solve_text(body: SolveTextRequest) -> StreamingResponse:
    """Stream a structured solution to a typed coding problem.

    Persists one `solve_sessions` row after the stream finishes;
    emits the new row id on the final `done` event so the UI can
    refresh its history drawer without a separate fetch.
    """
    started_ms = int(time.time() * 1000)

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse("meta", {"language": body.language or "python"})
        collected: list[str] = []
        try:
            async for chunk in code_solver.solve_text(body.problem, body.language):
                collected.append(chunk)
                yield _sse("token", {"text": chunk})
        except LLMError as exc:
            yield _sse("error", {"detail": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"detail": f"Unexpected error: {exc}"})
            return

        # Architecture.md §"Response architecture" — shape the
        # accumulated answer before persistence. Emits `artifacts`
        # when the answer is multi-file (Dockerfile + compose + …).
        full_response = "".join(collected)
        try:
            from app.core.config_loader import cfg as _cfg
            from app.response_arch import finalize as _finalize

            if _cfg.response_arch.enabled:
                shaped = _finalize(
                    full_response,
                    question=body.problem,
                    depth=_cfg.response_arch.default_depth,
                )
                full_response = shaped.text.strip() or full_response
                if shaped.artifacts:
                    yield _sse(
                        "artifacts",
                        {
                            "items": [
                                {
                                    "filename": a.filename,
                                    "language": a.language,
                                    "content": a.content,
                                }
                                for a in shaped.artifacts
                            ]
                        },
                    )
        except Exception:  # noqa: BLE001
            pass

        solve_id = await _persist_solve(
            description=body.problem,
            response=full_response,
            source="text",
            language=body.language,
            latency_ms=int(time.time() * 1000) - started_ms,
        )
        yield _sse("done", {"solve_id": solve_id})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/image")
async def solve_image(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    extra_context: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
    code_model: str | None = Form(default=None),
) -> StreamingResponse:
    """Stream a solution from a screenshot. Persists a `solve_sessions`
    row that carries the OCR-extracted problem text + the streamed
    answer, plus the saved image bytes' BlobStore path."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    # Stash the screenshot in BlobStore so the user can re-view it
    # from the history later (or the UI can re-display it inline).
    image_path: str | None = None
    try:
        blob_id = uuid.uuid4().hex
        ext = (file.filename or "screenshot.png").rsplit(".", 1)
        ext_suffix = f".{ext[-1]}" if len(ext) == 2 else ".png"
        image_path = await get_blobs().put(
            f"solve/{blob_id}{ext_suffix}", raw
        )
    except Exception:  # noqa: BLE001
        # BlobStore down → still solve, just don't keep the image.
        image_path = None

    started_ms = int(time.time() * 1000)

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse("meta", {
            "language": language or "python",
            "filename": file.filename,
            "vision_model": vision_model or "(default)",
            "code_model": code_model or "(default)",
        })

        collected: list[str] = []
        extracted_problem: str | None = None
        try:
            async for item in code_solver.solve_image(
                raw,
                language=language,
                extra_context=extra_context,
                vision_model=vision_model,
                code_model=code_model,
            ):
                if isinstance(item, code_solver.SolveStatus):
                    yield _sse("status", {"text": item.text})
                elif isinstance(item, code_solver.SolveExtracted):
                    # OCR'd problem text — captured for persistence and
                    # also surfaced to the UI so the user sees what was
                    # read before the answer streams.
                    extracted_problem = item.text
                    yield _sse("extracted", {"text": item.text})
                else:
                    collected.append(item)
                    yield _sse("token", {"text": item})
        except LLMError as exc:
            yield _sse("error", {"detail": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"detail": f"Unexpected error: {exc}"})
            return

        # Description: prefer the OCR'd problem (two-step pipeline);
        # otherwise fall back to mining the streamed response for its
        # "Problem" section so single-step solves are still searchable
        # in the history drawer instead of all reading as "(image solve)".
        response_text = "".join(collected)
        description = (
            extracted_problem
            or _extract_problem_from_response(response_text, file.filename)
        )
        # Response-architecture shaping (same as the text path).
        try:
            from app.core.config_loader import cfg as _cfg
            from app.response_arch import finalize as _finalize

            if _cfg.response_arch.enabled:
                shaped = _finalize(
                    response_text,
                    question=description,
                    depth=_cfg.response_arch.default_depth,
                )
                response_text = shaped.text.strip() or response_text
                if shaped.artifacts:
                    yield _sse(
                        "artifacts",
                        {
                            "items": [
                                {
                                    "filename": a.filename,
                                    "language": a.language,
                                    "content": a.content,
                                }
                                for a in shaped.artifacts
                            ]
                        },
                    )
        except Exception:  # noqa: BLE001
            pass
        solve_id = await _persist_solve(
            description=description,
            response=response_text,
            source="image",
            language=language,
            image_path=image_path,
            vision_model=vision_model,
            code_model=code_model,
            latency_ms=int(time.time() * 1000) - started_ms,
        )
        yield _sse("done", {"solve_id": solve_id})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- History --------------------------------------------------------------
@router.get("/sessions")
async def list_solve_sessions(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Return all Solve sessions, newest first.

    Plain dicts (not Pydantic) so the response surface tolerates
    whatever shape `row.id` happens to be — same approach as
    `/api/conversations` to avoid a UUID / Pydantic mismatch.
    """
    rows = await SolveRepo(session).list(user_id=get_default_user_id())
    return [
        {
            "id": str(r.id),
            "title": r.title,
            "source": r.source,
            "language": r.language,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/sessions/{solve_id}")
async def get_solve_session(
    solve_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """One solve with the full problem statement + response."""
    row = await SolveRepo(session).get(solve_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Solve session not found")
    return {
        "id": str(row.id),
        "title": row.title,
        "description": row.description,
        "response": row.response,
        "language": row.language,
        "source": row.source,
        "image_path": row.image_path,
        "vision_model": row.vision_model,
        "code_model": row.code_model,
        "latency_ms": row.latency_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


class SolvePatchBody(BaseModel):
    """PATCH body — only `title` is supported today."""
    title: str | None = None


@router.patch("/sessions/{solve_id}")
async def patch_solve_session(
    solve_id: str,
    body: SolvePatchBody,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Rename a solve session. Returns the updated row summary."""
    from storage.models import SolveSession

    if body.title is None:
        raise HTTPException(status_code=400, detail="`title` is required")
    new_title = body.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="`title` cannot be empty")
    try:
        row = await session.get(SolveSession, uuid.UUID(solve_id))
    except (TypeError, ValueError):
        row = None
    if row is None:
        raise HTTPException(status_code=404, detail="Solve session not found")
    row.title = new_title[:200]
    await session.commit()
    await session.refresh(row)
    return {
        "id": str(row.id),
        "title": row.title,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.delete("/sessions/{solve_id}")
async def delete_solve_session(
    solve_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Remove one solve from history. The blob (if any) is left on
    disk — orphaned blobs are cheap and a periodic GC job is the
    right place to clean them up."""
    ok = await SolveRepo(session).delete(solve_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Solve session not found")
    await session.commit()
    return {"ok": True}


@router.get("/sessions/{solve_id}/image")
async def get_solve_image(
    solve_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Stream the original screenshot bytes for one image solve.

    The Solve detail panel calls this to render the captured screen
    inline. Falls back to 404 when the row has no `image_path` or
    when the blob has gone missing from the BlobStore.
    """
    from fastapi.responses import Response as _Response

    row = await SolveRepo(session).get(solve_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Solve session not found")
    if not row.image_path:
        raise HTTPException(status_code=404, detail="No image stored for this solve")

    # `image_path` is the canonical reference the BlobStore returned
    # at upload time. FilesystemBlobs stores an absolute path; MinIO
    # stores an `s3://bucket/key` URL. The store's `get` knows how to
    # resolve either, but we need to pass it the relative key —
    # FilesystemBlobs.put() returned the absolute, so we strip the
    # root prefix back off.
    blobs = get_blobs()
    raw_path = row.image_path
    # Best-effort relative-key extraction. Works for both fs and minio
    # because both adapters tolerate prefix paths.
    try:
        bytes_ = await blobs.get(raw_path)
    except Exception:
        # Try stripping a `solve/{uuid}.ext` tail off the absolute path.
        from pathlib import Path as _P

        rel = _P(raw_path).name
        try:
            bytes_ = await blobs.get(f"solve/{rel}")
        except Exception:
            raise HTTPException(
                status_code=404, detail="Image blob missing from BlobStore"
            )

    # Sniff content-type from extension.
    ext = (raw_path.rsplit(".", 1)[-1] or "png").lower()
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return _Response(content=bytes_, media_type=mime)
