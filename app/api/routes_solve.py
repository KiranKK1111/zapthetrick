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
    # The OPEN chat conversation to record this exchange into (question +
    # answer as normal messages). None → recorded into a NEW conversation
    # whose id rides the `done` frame so the client adopts it.
    conversation_id: str | None = None


async def _persist_to_chat(conversation_id: str | None,
                           problem: str, answer: str) -> str | None:
    """Record the solve exchange into the user's CHAT history — the captured
    problem as the user message, the solution as the assistant message — so a
    Solve lives in the same conversation the user had open (reload-safe),
    exactly like a typed question. Creates a conversation when none is open.
    Fail-open: a DB hiccup never harms the already-streamed answer."""
    import logging
    log = logging.getLogger(__name__)
    if not (answer or "").strip():
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    try:
        from storage.models import Conversation, Message
        async with factory() as session:
            convo = None
            if conversation_id:
                try:
                    convo = await session.get(
                        Conversation, uuid.UUID(str(conversation_id)))
                except (ValueError, TypeError):
                    convo = None
            if convo is None:
                title = (" ".join((problem or "Solve")[:200].split()[:6])[:200]
                         or "Solve")
                convo = Conversation(title=title)
                session.add(convo)
                await session.flush()
            session.add(Message(conversation_id=convo.id, role="user",
                                content=(problem or
                                         "(screenshot problem)")[:16000]))
            session.add(Message(conversation_id=convo.id, role="assistant",
                                content=answer[:64000]))
            await session.commit()
            return str(convo.id)
    except Exception as exc:  # noqa: BLE001
        log.warning("solve chat persist skipped: %s", exc)
        return None


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

        # §3.5 toolchain prefetch: the target language is known upfront here, so
        # warm its runtime OFF the request path while the solution streams — the
        # verifier then starts hot. Best-effort; no-op unless enabled.
        try:
            from app.sandbox import pool as _pool
            if _pool.prefetch_enabled():
                _pool.prefetch_toolchain(body.language or "python")
        except Exception:  # noqa: BLE001
            pass

        # §3.2 problem-fingerprint cache: an identical problem+language already
        # solved is re-served instantly (no re-reasoning). Per-user scoped +
        # revalidated; a miss / disabled falls straight through to the solver.
        _fp_scope = ""
        try:
            from storage.context import get_request_user_id as _uid
            _fp_scope = str(_uid() or "")
        except Exception:  # noqa: BLE001
            _fp_scope = ""
        try:
            from app.solve import fingerprint as _fp
            _cached = await _fp.get(body.problem, body.language or "",
                                    scope=_fp_scope)
        except Exception:  # noqa: BLE001
            _cached = None
        if _cached:
            _step = 240
            for _ci in range(0, len(_cached), _step):
                yield _sse("token", {"text": _cached[_ci:_ci + _step]})
            solve_id = await _persist_solve(
                description=body.problem, response=_cached, source="text",
                language=body.language,
                latency_ms=int(time.time() * 1000) - started_ms)
            _cid = await _persist_to_chat(
                body.conversation_id, body.problem, _cached)
            yield _sse("done", {"solve_id": solve_id, "cached": True,
                                "conversation_id": _cid})
            return

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

        # MermaidDiagramVisualizations.md #1 (model half): re-derive a diagram as
        # structured IR and generate from that. Off by default — `plan_answer`
        # returns immediately then, so this costs nothing unless switched on. Runs
        # BEFORE the solution cache below so a cached hit replays the planned
        # diagram instead of re-planning it on every identical ask.
        try:
            from app.diagrams.lane import plan_answer as _plan_diagrams
            full_response = await _plan_diagrams(
                full_response, request=body.problem) or full_response
        except Exception:  # noqa: BLE001
            pass

        # §3.2 cache the fresh solution so the next identical ask is instant.
        try:
            from app.solve import fingerprint as _fp
            await _fp.put(body.problem, body.language or "", full_response,
                          scope=_fp_scope)
        except Exception:  # noqa: BLE001
            pass

        solve_id = await _persist_solve(
            description=body.problem,
            response=full_response,
            source="text",
            language=body.language,
            latency_ms=int(time.time() * 1000) - started_ms,
        )
        _cid = await _persist_to_chat(
            body.conversation_id, body.problem, full_response)
        yield _sse("done", {"solve_id": solve_id, "conversation_id": _cid})

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
    conversation_id: str | None = Form(default=None),
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

        # Stage-4 §3.2 — structured extraction (JSON contract) + language ladder
        # + problem-fingerprint cache, ahead of the reasoning step. Flag-gated
        # (`code_solver.structured_extraction`) + fail-open: any hiccup drops back
        # to today's delimited OCR path inside solve_image.
        _pre_extracted: str | None = None
        _pre_language: str | None = language
        _fp: str | None = None
        _served_from_cache = False
        try:
            from app.codeintel import solve_extract as _sx
            from app.codeintel import solve_fingerprint as _sfp
            if _sx.enabled():
                _got = await _sx.extract_structured(
                    raw, vision_model=vision_model, extra_context=extra_context)
                if _got is not None:
                    _lang, _src = _sx.resolve_language(_got, requested=language)
                    _pre_language = _lang or language
                    _pre_extracted = _got.to_delimited()
                    yield _sse("status", {"text": _got.summary()})
                    _fp = _sfp.fingerprint(_got.statement, _pre_language)
                    _cached = _sfp.get(_fp)
                    if _cached:
                        # An already-solved problem returns instantly.
                        extracted_problem = _pre_extracted
                        yield _sse("extracted", {"text": _pre_extracted})
                        collected.append(_cached)
                        yield _sse("token", {"text": _cached})
                        _served_from_cache = True
        except Exception:  # noqa: BLE001
            _pre_extracted, _fp = None, None

        if not _served_from_cache:
            try:
                async for item in code_solver.solve_image(
                    raw,
                    language=language,
                    extra_context=extra_context,
                    vision_model=vision_model,
                    code_model=code_model,
                    pre_extracted=_pre_extracted,
                    pre_language=_pre_language,
                ):
                    if isinstance(item, code_solver.SolveStatus):
                        yield _sse("status", {"text": item.text})
                    elif isinstance(item, code_solver.SolveExtracted):
                        # The extracted problem text — captured for persistence
                        # and surfaced to the UI before the answer streams.
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
            # Cache the fresh solution under the problem fingerprint (§3.2).
            if _fp and collected:
                try:
                    from app.codeintel import solve_fingerprint as _sfp2
                    _sfp2.put(_fp, "".join(collected))
                except Exception:  # noqa: BLE001
                    pass

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

        # MermaidDiagramVisualizations.md #1 (model half), same as the text path
        # above: `finalize` already ran the deterministic compile lane; this
        # re-derives a diagram as structured IR when that could not clean it up.
        # Off by default, so `plan_answer` returns immediately.
        try:
            from app.diagrams.lane import plan_answer as _plan_diagrams
            response_text = await _plan_diagrams(
                response_text, request=description) or response_text
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
        _cid = await _persist_to_chat(conversation_id, description,
                                      response_text)
        yield _sse("done", {"solve_id": solve_id, "conversation_id": _cid})

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


async def _owned_solve_or_404(session, solve_id):
    """Fetch a solve session and verify it belongs to the current user (§10.1c).
    Legacy NULL-owned rows stay accessible (pre-accounts data)."""
    from app.api.auth import resolve_user_id
    from storage.models import SolveSession
    try:
        row = await session.get(SolveSession, uuid.UUID(solve_id))
    except (TypeError, ValueError):
        row = None
    if row is None:
        raise HTTPException(status_code=404, detail="Solve session not found")
    uid = await resolve_user_id()
    if row.user_id is not None and uid is not None and row.user_id != uid:
        raise HTTPException(status_code=404, detail="Solve session not found")
    return row


@router.get("/sessions/{solve_id}")
async def get_solve_session(
    solve_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """One solve with the full problem statement + response."""
    row = await _owned_solve_or_404(session, solve_id)
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
    if body.title is None:
        raise HTTPException(status_code=400, detail="`title` is required")
    new_title = body.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="`title` cannot be empty")
    row = await _owned_solve_or_404(session, solve_id)
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
    await _owned_solve_or_404(session, solve_id)  # owner-only
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

    row = await _owned_solve_or_404(session, solve_id)
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
