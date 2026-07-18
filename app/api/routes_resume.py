"""Resume Q&A endpoints (persona mode + RAG).

  POST /api/resume/upload       multipart upload -> parse -> profile -> RAG ingest
  GET  /api/resumes             list uploaded resumes (most recent first)
  GET  /api/resume/{id}         one resume with its parsed profile
  POST /api/resume/ask/stream   SSE: orchestrator-driven first-person answer

The /ask endpoint goes through the full orchestrator: question
classification -> RAG retrieval -> persona answer streamed back as SSE.
Storage layer:
  - File bytes -> [BlobStore]                  (filesystem or MinIO)
  - Profile + raw_text + metadata -> Postgres  (via [ResumeRepo])
  - Chunks -> Postgres + [VectorStore]          (Qdrant or Chroma)
"""
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_loader import cfg
from app.core.llm_client import LLMError
from app.documents import progress as _progress
from app.core.orchestrator import AnswerContext, answer_question
from app.persona.profile_builder import build_profile
from app.rag.ingest import IngestWarning, ingest_resume
from app.resume_parser import UnsupportedResumeFormat, extract_text
from app.schemas import (
    ResumeAskRequest,
    ResumeDetail,
    ResumeSummary,
    ResumeUploadResponse,
)
from storage import get_session
from storage.db import get_session_factory
from storage.blobs import get_blobs
from storage.device import ensure_device_user
from storage.repos import ResumeRepo
from storage.users import get_default_user_id


router = APIRouter(prefix="/api")

# Strong refs to the fire-and-forget profile-extraction tasks. asyncio only
# keeps a WEAK reference to a bare create_task(), so without this the GC can
# collect the task mid-run — which is exactly why the resume "sometimes" got
# detected and sometimes didn't. Holding the task here guarantees it finishes.
_BG_TASKS: set = set()


def _profile_dict(profile) -> dict:
    """Coerce a stored profile back into a dict.

    `profile` is JSONB in the schema so we usually get a dict already;
    the string branch is defensive against rows written by a pre-JSONB
    code path.
    """
    if isinstance(profile, dict):
        return profile
    if isinstance(profile, str) and profile:
        try:
            return json.loads(profile)
        except json.JSONDecodeError:
            return {}
    return {}


def _placeholder_profile(text: str, filename: str) -> dict:
    """Cheap profile we drop into the row before the LLM extractor runs.

    The chat path uses `profile["summary"]` as a fallback when the
    structured fields are missing — so this keeps the Resume tab fully
    functional during the seconds-to-minutes before the background
    extractor finishes.
    """
    return {
        "name": None,
        "headline": None,
        "years_experience": None,
        "current_role": None,
        "summary": (text or "")[:1500],
        "skills": [],
        "work_history": [],
        "education": [],
        "projects": [],
        # Flag the UI can render: "Analyzing…". Cleared when background
        # extraction completes.
        "_analyzing": True,
    }


async def _finalize_profile_in_background(
    resume_id: uuid.UUID, text: str
) -> None:
    """Run `build_profile` after the upload route has returned, then
    UPDATE the Resume row.

    Failures are logged but never visible — the placeholder profile
    keeps the app usable. Network / quota errors just leave
    `_analyzing` flipped to False without enriching the fields.
    """
    import logging
    from storage.db import get_session_factory
    from storage.models import Resume

    _log = logging.getLogger(__name__)
    try:
        profile = await build_profile(text)
        profile["_analyzing"] = False
    except Exception as exc:  # noqa: BLE001
        _log.warning("background profile extraction failed: %s", exc)
        # build_profile is meant to never raise, but if something unexpected
        # does, still detect the candidate from the text rather than going
        # blank — and stop the UI spinner.
        from app.persona.profile_builder import _heuristic_profile

        profile = _heuristic_profile(text)
        profile["_analyzing"] = False

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as s:
            row = await s.get(Resume, resume_id)
            if row is None:
                return
            row.profile = profile
            # Bump display name once we have a real name from the LLM.
            name = profile.get("name") if isinstance(profile, dict) else None
            if isinstance(name, str) and name.strip():
                row.display_name = name.strip()
            await s.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("background profile commit failed: %s", exc)
        return
    # Latency batch 2026-07-11 (#2): with the profile ready, pre-generate
    # grounded answers for the common interview questions in the background —
    # live matches then stream them instantly. Fire-and-forget; failures only
    # mean the live path generates normally.
    try:
        import asyncio as _aio

        from app.live import prepared as _prepared
        if _prepared.enabled() and isinstance(profile, dict):
            _t = _aio.create_task(
                _prepared.prepare_for_resume(str(resume_id), profile))
            _BG_PREPARED.add(_t)
            _t.add_done_callback(_BG_PREPARED.discard)
    except Exception:  # noqa: BLE001
        pass


# GC guard for the fire-and-forget prepared-answer tasks.
_BG_PREPARED: set = set()


# ---- Upload ------------------------------------------------------------


async def _ingest_in_background(resume_id: uuid.UUID, text: str) -> None:
    """Run the RAG ingest (chunk + embed + Qdrant) after the upload route has
    returned, on its own DB session. Makes resume-against-chat RAG ready a
    beat after the upload without blocking the user-visible response.

    Failures are logged but never surfaced — the resume is already usable for
    Live/persona grounding (which reads the profile JSON, not the vectors)."""
    import logging
    from storage.db import get_session_factory

    _log = logging.getLogger(__name__)
    rid = str(resume_id)
    factory = get_session_factory()
    if factory is None:
        _progress.fail(rid, "Database not ready — ingest skipped")
        return
    try:
        async with factory() as s:
            try:
                n = await ingest_resume(rid, text, s)
                await s.commit()
                _progress.finish(
                    rid,
                    detail=(f"Ready — {n} chunks indexed" if n
                            else "Ready — nothing to index"),
                )
            except IngestWarning as warn:
                # Chunks are durable in Postgres; only the vector index is
                # degraded. The resume is usable — report done, not error.
                await s.commit()
                _progress.finish(rid, detail=f"Ready (vector index degraded: {warn})")
    except Exception as exc:  # noqa: BLE001
        _log.warning("background resume ingest failed: %s", exc)
        _progress.fail(rid, f"Processing failed: {exc}")
@router.post("/resume/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
) -> ResumeUploadResponse:
    """Parse + ingest a resume. Returns IMMEDIATELY after RAG ingest;
    the structured profile is filled in by a background task.

    `session_id` (optional) links the resume to a live-interview session, so
    the resume "belongs to" that session and reloads with it.

    Why deferred: the LLM call inside `build_profile` is the slowest
    step in the whole app (5-30s on free OpenRouter). Doing it
    synchronously in the route made the upload feel hung. Now:

      1. Parse PDF/DOCX                          (fast)
      2. Save bytes to BlobStore                 (fast)
      3. Insert row with a placeholder profile   (fast)
      4. RAG ingest — chunk + embed + Qdrant     (a couple seconds)
      5. Return HTTP 200                          <-- user sees the resume
      6. (background) build_profile + UPDATE row  <-- fills in name/etc.

    The Flutter client can poll `GET /api/resume/{id}` to see the
    structured fields land, or just refresh — either works.
    """
    from app.documents.parser import MAX_UPLOAD_BYTES

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large "
                f"({file.size / 1024 / 1024:.0f} MB). The limit is "
                f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
            ),
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large "
                f"({len(file_bytes) / 1024 / 1024:.0f} MB). The limit is "
                f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
            ),
        )

    try:
        # Worker thread: pdfplumber/docx parsing is CPU-bound and previously
        # blocked the event loop for large PDFs (upload-slowness report
        # 2026-07-08) — freezing SSE keepalives and other requests with it.
        import asyncio as _aio
        text = await _aio.to_thread(
            extract_text, file_bytes, file.filename or "resume")
    except UnsupportedResumeFormat as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"Could not parse resume: {exc}"
        )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text extracted from the file. Is it an image-only PDF?",
        )

    # 1. Placeholder profile so the Resume tab has something to render
    #    immediately. Replaced by the background task in 10-30s.
    placeholder = _placeholder_profile(text, file.filename or "resume")
    display_name = file.filename or "Resume"

    # 2. Write the raw bytes to the BlobStore.
    blob_id = uuid.uuid4().hex
    blob_path = f"resumes/{blob_id}_{file.filename or 'resume'}"
    file_path = await get_blobs().put(blob_path, file_bytes)

    # 3. Insert the row. Resolve the device user EAGERLY here —
    # `get_default_user_id()` returns None until bootstrap's background
    # migration task has finished `ensure_device_user`. Uploading in
    # that window would otherwise save the resume with `user_id=NULL`,
    # which later list queries (filtered by the now-known device UUID)
    # silently exclude — and the user sees "my resume is gone".
    user_id = get_default_user_id() or await ensure_device_user()
    repo = ResumeRepo(session)
    resume = await repo.create(
        user_id=user_id,
        filename=file.filename or "resume",
        file_path=file_path,
        display_name=display_name,
        profile=placeholder,
        raw_text=text,
        embedding_model=cfg.embeddings.model,
        active=True,
    )
    await repo.mark_active(resume.id)
    await session.flush()

    # 4. RAG ingest (chunk + embed + Qdrant) is deferred to the background.
    #    It loads the BGE-m3 embedder + reranker on CPU, which is the slow
    #    part of an upload — and Live/persona grounding reads the profile
    #    JSON, not the vectors, so the upload need not block on it. Resume-
    #    against-chat RAG becomes ready a beat later (the row already exists).
    await session.commit()
    await session.refresh(resume)

    # Link the resume to the live session that uploaded it, so it "belongs to"
    # that session and reloads with it. Best-effort — a bad/non-live id is
    # ignored.
    if session_id:
        try:
            from storage.repos import SessionRepo
            await SessionRepo(session).set_resume(session_id, resume.id)
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()

    # Progress registry: the synchronous steps (upload + parse) are already
    # done by the time the client learns the resume_id, so seed the entry
    # past them; the background ingest advances chunk -> embed -> index.
    _progress.begin(str(resume.id), op="upload")
    _progress.set_stage(
        str(resume.id),
        "parse",
        fraction=1.0,
        detail=f"Extracted {len(text)} characters",
        counts={"characters": len(text)},
    )

    # 5. Fire-and-forget the heavy work: RAG ingest + structured profile
    #    extraction. The route returns the placeholder immediately; the row
    #    UPDATEs as each finishes.
    import asyncio as _asyncio

    for coro, tag in (
        (_ingest_in_background(resume.id, text), "ingest"),
        (_finalize_profile_in_background(resume.id, text), "profile-extract"),
    ):
        _task = _asyncio.create_task(coro, name=f"{tag}-{resume.id}")
        _BG_TASKS.add(_task)
        _task.add_done_callback(_BG_TASKS.discard)

    return ResumeUploadResponse(
        resume_id=str(resume.id),
        display_name=resume.display_name,
        filename=resume.filename,
        profile=_profile_dict(resume.profile),
        created_at=resume.uploaded_at,
    )


# ---- List & detail -----------------------------------------------------
async def _claim_orphans(session: AsyncSession, user_id) -> None:
    """Adopt any `user_id IS NULL` resumes to the current device user.

    The orphan rows come from one of two cases:
      1. An upload landed during the migration window when
         `_cached_user_id` was still None.
      2. The device user was created *after* a fresh install where
         legacy rows pre-date the `users` table.

    Either way: in single-user-on-device mode every resume must
    belong to *this* user. Reattaching is safe and idempotent.
    """
    if user_id is None:
        return
    from sqlalchemy import update

    from storage.models import Resume

    await session.execute(
        update(Resume).where(Resume.user_id.is_(None)).values(user_id=user_id)
    )
    await session.commit()


@router.get("/resumes", response_model=list[ResumeSummary])
async def list_resumes(
    session: AsyncSession = Depends(get_session),
) -> list[ResumeSummary]:
    # Resolve the device user eagerly — same reasoning as upload: the
    # cache might still be None right after a hot-reload / re-dock.
    user_id = get_default_user_id() or await ensure_device_user()
    # Self-heal: re-parent any orphan rows so they show up below.
    await _claim_orphans(session, user_id)
    rows = await ResumeRepo(session).list(user_id=user_id)
    return [
        ResumeSummary(
            id=str(r.id),
            display_name=r.display_name,
            filename=r.filename,
            created_at=r.uploaded_at,
        )
        for r in rows
    ]


@router.get("/resume/{resume_id}", response_model=ResumeDetail)
async def get_resume(
    resume_id: str,
    session: AsyncSession = Depends(get_session),
) -> ResumeDetail:
    resume = await ResumeRepo(session).get(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found")
    return ResumeDetail(
        id=str(resume.id),
        display_name=resume.display_name,
        filename=resume.filename,
        profile=_profile_dict(resume.profile),
        created_at=resume.uploaded_at,
    )


@router.get("/resumes/{resume_id}/progress")
async def resume_progress(resume_id: str) -> dict:
    """Poll staged progress for an in-flight upload or delete.

    Returns the live registry entry (stage / percent / counts / done /
    error). If nothing is (or was) tracked for this id — e.g. the server
    restarted mid-poll — returns a `{"stage": "unknown"}` stub with 200,
    so a poller can distinguish "no tracking" from a transport error
    without special-casing 404s.
    """
    entry = _progress.get(resume_id)
    if entry is None:
        return {
            "stage": "unknown",
            "stage_index": 0,
            "total_stages": 0,
            "stages": [],
            "percent": 0.0,
            "detail": "",
            "counts": {},
            "done": False,
            "error": None,
            "op": None,
        }
    return entry


@router.delete("/resume/{resume_id}")
async def delete_resume(
    resume_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a resume: its vectors, BM25 cache, chunks, blob, and the row.

    Vector / blob cleanup is best-effort (Postgres is authoritative) — the
    row removal is what makes the resume disappear from the list and stop
    grounding answers."""
    repo = ResumeRepo(session)
    resume = await repo.get(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found")
    file_path = resume.file_path

    _progress.begin(resume_id, op="delete")
    try:
        # 1. Best-effort: drop the vector collection + BM25 cache.
        _progress.set_stage(resume_id, "vectors", detail="Removing vector index")
        from app.rag import retriever, store
        try:
            await store.delete_resume(resume_id)
        except Exception:  # noqa: BLE001 — vector store offline; Postgres is authoritative
            pass
        try:
            retriever.invalidate_bm25(resume_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            from app.live import prepared as _prepared
            _prepared.drop(resume_id)
        except Exception:  # noqa: BLE001
            pass

        # 2. Clear chunks, then the row.
        _progress.set_stage(resume_id, "chunks", detail="Removing chunk rows")
        await repo.replace_chunks(resume_id, [])
        # Null out any session that points at this resume so none dangle.
        try:
            from storage.repos import SessionRepo
            await SessionRepo(session).clear_resume_everywhere(resume_id)
        except Exception:  # noqa: BLE001
            pass
        _progress.set_stage(resume_id, "row", detail="Removing resume record")
        await repo.delete(resume_id)

        # 3. Best-effort blob removal.
        if file_path:
            try:
                await get_blobs().delete(file_path)
            except Exception:  # noqa: BLE001
                pass

        await session.commit()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _progress.fail(resume_id, f"Delete failed: {exc}")
        raise
    _progress.finish(resume_id, detail="Resume deleted")
    return {"ok": True, "deleted": resume_id}


# ---- Re-index: rebuild Qdrant vectors from the Postgres chunks ----------
@router.post("/resume/{resume_id}/reindex")
async def reindex_resume(
    resume_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Rebuild this resume's vector index from the durable Postgres chunks.

    Use when the vector store has lost or never received the
    embeddings — e.g. Qdrant was wiped, the storage volume rolled
    back, or the resume was uploaded while Qdrant was unreachable.
    Postgres is the source of truth; this re-runs `embedder.embed`
    over the existing `resume_chunks.content` and upserts the
    vectors back, **without** asking the user to re-upload the PDF.
    """
    import asyncio as _asyncio

    from app.rag import embedder
    from app.rag import store as vec_store
    from app.rag.retriever import invalidate_bm25

    repo = ResumeRepo(session)
    resume = await repo.get(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found")

    chunks = await repo.fetch_chunks(resume_id)
    if not chunks:
        return {
            "ok": True,
            "reindexed": 0,
            "note": "No chunks on file — re-upload the resume.",
        }

    # Drop any stale Qdrant collection first so a partial old index
    # can't bleed into the new one. invalidate_bm25 is a kept-for-
    # compat no-op (Postgres FTS is server-side now).
    try:
        await vec_store.delete_resume(str(resume.id))
    except Exception:
        pass
    invalidate_bm25(str(resume.id))

    texts = [c.content for c in chunks]
    embeddings = await _asyncio.to_thread(embedder.embed, texts)
    metadatas = [
        {
            "resume_id": str(resume.id),
            "chunk_id": str(c.id),
            "position": c.position,
            "section": c.section_type or "",
        }
        for c in chunks
    ]
    ids = [str(c.vector_point_id or c.id) for c in chunks]

    try:
        await vec_store.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Vector store rejected reindex: {exc}",
        )
    return {"ok": True, "reindexed": len(chunks)}


# ---- Ask AS the candidate (orchestrator-backed) ------------------------
def _json_default(o):
    """Handle UUID / datetime in SSE payloads — Postgres returns UUIDs
    for primary keys and `json.dumps` doesn't know how to serialize
    them. Without this hook the SSE generator would raise mid-stream
    and the client would see "Connection closed while receiving data"."""
    import uuid as _uuid
    from datetime import datetime as _dt

    if isinstance(o, _uuid.UUID):
        return str(o)
    if isinstance(o, _dt):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=_json_default)}\n\n"


@router.post("/resume/ask/stream")
async def ask_as_candidate(
    body: ResumeAskRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Stream a first-person answer via the orchestrator."""
    resume = await ResumeRepo(session).get(body.resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found")

    profile = _profile_dict(resume.profile)
    if not profile:
        profile = {"summary": (resume.raw_text or "")[:2000]}

    resume_id = str(resume.id)
    display_name = resume.display_name
    # Per-request session id — keeps the context tracker independent
    # across browser tabs. Long-lived sessions should pass session_id
    # in via the WebSocket route instead.
    session_id = body.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        yield _sse(
            "start",
            {"resume_id": resume_id, "display_name": display_name, "session_id": session_id},
        )
        # Late-bind via the helper — migrations run in the background,
        # so SessionFactory imported at module load could still be None
        # when this generator finally runs.
        factory = get_session_factory()
        if factory is None:
            yield _sse(
                "error",
                {
                    "detail": (
                        "Database not ready — migrations are still running "
                        "or Postgres is unreachable. Check Settings -> Database."
                    )
                },
            )
            return
        async with factory() as db_session:
            ctx = AnswerContext(
                question=body.question,
                session_id=session_id,
                profile=profile,
                resume_id=resume_id,
                db_session=db_session,
                # Skip the classifier LLM round-trip — questions hitting
                # the Resume tab are 95%+ behavioral, and even when
                # they're not, "behavioral" still routes through the
                # retrieval + persona path the user expects. Saves
                # ~500ms-3s per turn on OpenRouter's free tier.
                forced_type="behavioral",
            )
            async for event in answer_question(ctx):
                yield _sse(event.kind, event.data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
