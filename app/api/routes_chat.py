"""
Generic chat endpoints (health, conversation list/detail, patch, delete).

Endpoints:
  GET    /api/health                 -- backend, database, and LLM status
  GET    /api/conversations          -- list conversation summaries
  GET    /api/conversations/{id}     -- one conversation with all messages
  PATCH  /api/conversations/{id}     -- rename / pin / archive / tag
  DELETE /api/conversations/{id}     -- delete a conversation + its blobs/vectors

Message streaming lives in routes_agents (/api/agents/stream) and
routes_attachments (/api/chat/upload-stream); the old /api/chat/stream surface
was unused by the client and has been removed.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_loader import cfg
from app.core.llm_client import llm
from app.database import Conversation, Message, get_session
from app.schemas import HealthResponse

router = APIRouter(prefix="/api")


# ---- Health -------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    """Report status of the backend, database, and the configured LLM."""
    try:
        await session.execute(select(Conversation).limit(1))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001 -- surface any failure
        db_status = f"error: {exc}"

    llm_status = await llm.health()
    return HealthResponse(
        database=db_status,
        llm=llm_status.get("status", "unknown"),
        provider=llm_status.get("provider", cfg.llm.provider),
        model=llm_status.get("model", cfg.llm.model),
    )


# ---- Conversations ------------------------------------------------------
import asyncio
import logging
import uuid as _uuid

_log = logging.getLogger(__name__)

# Detached background tasks (artifact cleanup after a delete) — kept referenced
# so the event loop doesn't GC them mid-flight.
_BG_TASKS: set = set()


async def _cleanup_conversation_artifacts(
    conversation_id: str, blob_paths: list[str]
) -> None:
    """Slow, correctness-irrelevant cleanup of everything a deleted conversation
    OWNED — stored blobs (disk) + RAG vectors (Qdrant). Run detached so the
    delete endpoint returns as soon as the DB rows are committed. Fully
    best-effort: a failure here only leaves an orphaned artifact, never a
    half-deleted conversation."""
    if blob_paths:
        try:
            from storage.blobs import get_blobs

            store = get_blobs()
            for p in blob_paths:
                try:
                    await store.delete(p)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    if conversation_id:
        try:
            from app.rag.documents import drop_chat_collection

            await drop_chat_collection(str(conversation_id))
        except Exception:  # noqa: BLE001
            pass


def _schedule_artifact_cleanup(conversation_id: str, blob_paths: list[str]) -> None:
    """Fire-and-forget the owned-artifact cleanup for one (or many) deleted
    conversations without blocking the response."""
    try:
        t = asyncio.create_task(
            _cleanup_conversation_artifacts(conversation_id, blob_paths)
        )
        _BG_TASKS.add(t)
        t.add_done_callback(_BG_TASKS.discard)
    except RuntimeError:  # no running loop (shouldn't happen in a request)
        pass


def _iso_utc(dt) -> str | None:
    """Serialize a datetime as an ISO string that ALWAYS carries a timezone.
    A naive value (e.g. from `datetime.utcnow` written via `onupdate`, or a
    `timestamp without time zone` column) is treated as UTC and stamped with a
    `+00:00` offset — without it the Flutter client parses the string as LOCAL
    time and a just-updated chat shows a multi-hour-old relative timestamp
    (the "5h ago on a brand-new chat" bug at UTC+5:30)."""
    if dt is None:
        return None
    from datetime import timezone as _tz
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=_tz.utc)
    return dt.isoformat()


def _conversation_summary_dict(row) -> dict:
    """Build a flat dict the JSON encoder can serialize directly.

    Bypasses Pydantic's from_attributes path so a UUID primary key (or
    any other non-string id) can't trip the response validator. Whatever
    `row.id` is — UUID, string, integer — gets stringified.

    Carries the chat-history affordances (pinned/archived/tags/etc.) so
    the Flutter drawer can render them inline without a second fetch.
    """
    return {
        "id": str(row.id),
        "title": row.title or "",
        "updated_at": _iso_utc(row.updated_at),
        "pinned": bool(getattr(row, "pinned", False)),
        "archived": bool(getattr(row, "archived", False)),
        "tags": list(getattr(row, "tags", []) or []),
        "message_count": int(getattr(row, "message_count", 0) or 0),
        "last_message_at": _iso_utc(getattr(row, "last_message_at", None)),
        # The project this conversation belongs to (None = ungrouped). Lets the
        # sidebar show grouped chats ONLY under their project, not in the flat
        # Conversations list.
        "project_id": (
            str(row.project_id) if getattr(row, "project_id", None) else None
        ),
    }


def _envelope_for(row) -> dict | None:
    """The persisted response.v1 envelope (Architecture.md §5), or a minimal
    reconstruction from the row's columns for older rows that predate it. Only
    assistant turns carry an envelope. Fail-open → None on any error."""
    if getattr(row, "role", None) != "assistant":
        return None
    env = getattr(row, "envelope", None)
    if isinstance(env, dict) and env:
        return {**env, "message_id": str(row.id)}
    try:
        from app.response_arch.envelope import build_envelope
        src = row.sources if isinstance(getattr(row, "sources", None), dict) else {}
        return build_envelope(
            message_id=str(row.id),
            intent={"type": row.intent} if getattr(row, "intent", None) else None,
            incomplete=bool(getattr(row, "incomplete", False)),
            document=src if src.get("document") else None,
            model=getattr(row, "model", None),
        )
    except Exception:  # noqa: BLE001 — envelope is additive, never fatal
        return None


def _message_dict(row, feedback: str | None = None) -> dict:
    return {
        "id": str(row.id),
        "role": row.role,
        "content": row.content,
        # Prefer the full intent label stored in `sources` (live answers — never
        # truncated); fall back to the String(50) `intent` column (chat rows).
        "intent": (
            (row.sources or {}).get("intent")
            if isinstance(getattr(row, "sources", None), dict)
            and (row.sources or {}).get("intent")
            else getattr(row, "intent", None)
        ),
        # Live seniority-calibration ({band, track?, target?}) persisted in
        # `sources` — reloaded so the answer's calibration pill still renders.
        "calibration": (
            (row.sources or {}).get("calibration")
            if isinstance(getattr(row, "sources", None), dict)
            else None
        ),
        "created_at": (row.created_at.isoformat() if row.created_at else None),
        # Latest 👍/👎 the user left on this message (thumb_up | thumb_down | None)
        # so the chat UI can restore the like/dislike state on reload.
        "feedback": feedback,
        # Filenames the user attached to this turn (stored in `sources`).
        "attachments": (
            (row.sources or {}).get("attachments", [])
            if isinstance(getattr(row, "sources", None), dict)
            else []
        ),
        # Persisted image refs [{name, path}] so a retry/reload can re-attach
        # the image (fetched from /api/chat/attachment-image?path=...).
        "image_refs": (
            (row.sources or {}).get("images", [])
            if isinstance(getattr(row, "sources", None), dict)
            else []
        ),
        # Persisted document refs [{name, path}] — previewed from /api/blob.
        "file_refs": (
            (row.sources or {}).get("files", [])
            if isinstance(getattr(row, "sources", None), dict)
            else []
        ),
        # Hidden model directive (e.g. the Solve language instruction) so a
        # retry after reload re-passes it — the bubble text stays `content`.
        "instruction": (
            (row.sources or {}).get("instruction")
            if isinstance(getattr(row, "sources", None), dict)
            else None
        ),
        # True when this assistant turn was cut short (disconnect / Stop /
        # provider drop) — the UI offers Continue / Retry.
        "incomplete": bool(getattr(row, "incomplete", False)),
        # True only when the user explicitly asked to generate a document —
        # drives the inline document card / preview panel (persisted in
        # `sources.document`). NOT set for plain answers/summaries.
        "is_document": bool(
            (row.sources or {}).get("document")
            if isinstance(getattr(row, "sources", None), dict)
            else False
        ),
        # The format the document was generated in (pdf/docx/…), so the card's
        # Download button saves THAT format directly — no format dropdown.
        "document_format": (
            (row.sources or {}).get("format", "pdf")
            if isinstance(getattr(row, "sources", None), dict)
            else "pdf"
        ),
        # All requested formats when the user asked for several at once (e.g.
        # "a text and a markdown document") → one download card per format.
        # Defaults to the single [document_format] for older / single rows.
        "document_formats": (
            (row.sources or {}).get(
                "formats",
                [(row.sources or {}).get("format", "pdf")],
            )
            if isinstance(getattr(row, "sources", None), dict)
            else ["pdf"]
        ),
        # The project's name (for a ZIP) → used as the download filename / title.
        "document_name": (
            (row.sources or {}).get("project_name")
            if isinstance(getattr(row, "sources", None), dict)
            else None
        ),
        # Phase 4 — an agentic build/edit turn produced a modified workspace;
        # this is the download URL for its zip (so a reloaded run still offers
        # "Download project"). Null for ordinary turns.
        "workspace_download": (
            (row.sources or {}).get("download")
            if isinstance(getattr(row, "sources", None), dict)
            and (row.sources or {}).get("workspace")
            else None
        ),
        # Phase 8 — quality & trust: confidence band, provenance, red-team
        # review (persisted in `sources`) so a reloaded agent run keeps them.
        "agent_confidence": (
            (row.sources or {}).get("confidence")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        "agent_provenance": (
            (row.sources or {}).get("provenance") or []
            if isinstance(getattr(row, "sources", None), dict) else []
        ),
        "agent_review": (
            (row.sources or {}).get("review") or []
            if isinstance(getattr(row, "sources", None), dict) else []
        ),
        "agent_metrics": (
            (row.sources or {}).get("metrics")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        "agent_cross_verify": (
            (row.sources or {}).get("cross_verify")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        "agent_semantic_diff": (
            (row.sources or {}).get("semantic_diff") or []
            if isinstance(getattr(row, "sources", None), dict) else []
        ),
        "agent_tests": (
            (row.sources or {}).get("tests")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        "agent_git": (
            (row.sources or {}).get("git")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        "agent_security": (
            (row.sources or {}).get("security")
            if isinstance(getattr(row, "sources", None), dict) else None
        ),
        # Claims the Grounder flagged as unsupported by the attached evidence
        # (empty/absent = grounded). Drives a "⚠ unverified" chip on the bubble.
        "grounding": (
            (row.sources or {}).get("grounding", [])
            if isinstance(getattr(row, "sources", None), dict)
            else []
        ),
        # Unified response.v1 envelope (Architecture.md §5): the persisted object
        # so a reload matches what streamed live. Older rows (no stored envelope)
        # get a minimal one reconstructed from the other columns.
        "envelope": _envelope_for(row),
    }


@router.get("/conversations")
async def list_conversations(
    archived: bool | None = False,
    tag: str | None = None,
    q: str | None = None,
    type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Return conversations for the drawer.

    Query params:
      archived  False (default), True, or null for "all"
      tag       restrict to sessions whose tag array contains this label
      q         full-text search across titles + messages
      limit     max rows (default 100)
    """
    from storage.repos import SessionRepo

    try:
        repo = SessionRepo(session)
        if q:
            rows = await repo.search(q, limit=limit)
            if type is not None:
                rows = [r for r in rows if r.type == type]
        else:
            rows = await repo.list(
                type=type,
                archived=archived,
                tag=tag,
                limit=limit,
            )
        _log.info(
            "list_conversations: q=%r tag=%r archived=%r → %d row(s)",
            q,
            tag,
            archived,
            len(rows),
        )
        return [_conversation_summary_dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001 — surface for the UI
        _log.exception("list_conversations failed")
        raise HTTPException(
            status_code=500, detail=f"list_conversations: {exc.__class__.__name__}: {exc}"
        )


@router.get("/conversations/_debug")
async def conversations_debug(
    session: AsyncSession = Depends(get_session),
):
    """Diagnostic view: search_path + per-schema row counts.

    If the Chat history sidebar is empty but pgAdmin shows rows, hit
    this endpoint to see where SQLAlchemy is actually looking.
    """
    from sqlalchemy import text

    try:
        sp = await session.execute(text("SHOW search_path"))
        search_path = sp.scalar_one_or_none() or "(unknown)"

        # Count rows in every `sessions` table the user might have,
        # across all schemas, so we can spot the mismatch.
        per_schema = await session.execute(
            text(
                "SELECT table_schema, "
                "(xpath('/row/c/text()', query_to_xml("
                "'SELECT COUNT(*) AS c FROM '||"
                "quote_ident(table_schema)||'.sessions', true, false, '''')))[1]::text::int AS count "
                "FROM information_schema.tables "
                "WHERE table_name = 'sessions'"
            )
        )
        counts = [{"schema": r[0], "count": r[1]} for r in per_schema.all()]

        # Same for messages.
        per_schema_msgs = await session.execute(
            text(
                "SELECT table_schema, "
                "(xpath('/row/c/text()', query_to_xml("
                "'SELECT COUNT(*) AS c FROM '||"
                "quote_ident(table_schema)||'.messages', true, false, '''')))[1]::text::int AS count "
                "FROM information_schema.tables "
                "WHERE table_name = 'messages'"
            )
        )
        msg_counts = [{"schema": r[0], "count": r[1]} for r in per_schema_msgs.all()]

        return {
            "search_path": search_path,
            "sessions_per_schema": counts,
            "messages_per_schema": msg_counts,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {exc}"}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    limit: int | None = None,
    before: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Return one conversation with its messages (newest-window pagination).

    - No `limit`: every message (back-compat).
    - `limit=N`: the most recent N messages (chronological order), plus
      `has_more=true` when older messages exist beyond the window.
    - `before=<created_at ISO>`: only messages strictly older than the cursor
      (combine with `limit` to page backwards as the user scrolls up).
    """
    try:
        # session.get on a UUID-PK model needs the UUID object on some
        # asyncpg / SQLAlchemy combos — explicit coercion is safer than
        # relying on driver auto-parsing.
        try:
            key: object = _uuid.UUID(conversation_id)
        except (TypeError, ValueError):
            key = conversation_id

        convo = await session.get(Conversation, key)
        if convo is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Pull the messages explicitly (don't rely on lazy-loading the
        # relationship — async sessions don't expire-load).
        base_q = select(Message).where(Message.conversation_id == convo.id)
        if before:
            from datetime import datetime as _dt
            try:
                base_q = base_q.where(
                    Message.created_at < _dt.fromisoformat(before)
                )
            except ValueError:
                pass  # bad cursor → ignore, treat as no filter

        has_more = False
        if limit and limit > 0:
            # Fetch the newest `limit` (+1 to detect older), then flip back to
            # chronological order for display.
            rows = (
                await session.execute(
                    base_q.order_by(Message.created_at.desc()).limit(limit + 1)
                )
            ).scalars().all()
            has_more = len(rows) > limit
            messages = list(reversed(rows[:limit]))
        else:
            messages = (
                await session.execute(base_q.order_by(Message.created_at))
            ).scalars().all()

        # Latest feedback signal per message (most recent wins), so the chat
        # UI can restore 👍/👎 state when the conversation is reopened.
        from app.database import Feedback as _Feedback

        fb_rows = (
            await session.execute(
                select(_Feedback)
                .where(_Feedback.message_id.in_([m.id for m in messages] or [None]))
                .order_by(_Feedback.created_at)
            )
        ).scalars().all()
        fb_map: dict = {}
        for fb in fb_rows:
            fb_map[fb.message_id] = fb.signal  # later rows overwrite → latest wins

        return {
            "id": str(convo.id),
            "title": convo.title or "",
            "created_at": (convo.started_at.isoformat() if getattr(convo, "started_at", None) else None),
            "updated_at": (convo.updated_at.isoformat() if convo.updated_at else None),
            "messages": [_message_dict(m, fb_map.get(m.id)) for m in messages],
            "has_more": has_more,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _log.exception("get_conversation(%s) failed", conversation_id)
        raise HTTPException(
            status_code=500, detail=f"get_conversation: {exc.__class__.__name__}: {exc}"
        )


# ---- Conversation mutations --------------------------------------------
from pydantic import BaseModel as _BaseModel


class ConversationPatch(_BaseModel):
    """Partial-update body for PATCH /api/conversations/{id}.

    All fields optional. The repo layer skips unset values.
    """
    title: str | None = None
    pinned: bool | None = None
    archived: bool | None = None
    tags: list[str] | None = None


@router.patch("/conversations/{conversation_id}")
async def patch_conversation(
    conversation_id: str,
    body: ConversationPatch,
    session: AsyncSession = Depends(get_session),
):
    """Pin / archive / rename / retag one conversation in a single call."""
    from storage.repos import SessionRepo

    repo = SessionRepo(session)
    row = await repo.set_flags(
        conversation_id,
        pinned=body.pinned,
        archived=body.archived,
        title=body.title,
        tags=body.tags,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return _conversation_summary_dict(row)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Remove a conversation, its messages, AND everything it owns — the stored
    blobs (uploaded files/images + generated artifacts) and its RAG vectors — so
    nothing is left orphaned in Postgres."""
    from storage.repos import SessionRepo

    # Collect blob paths from this conversation's messages BEFORE the cascade
    # delete removes the rows.
    blob_paths: list[str] = []
    try:
        rows = (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        ).scalars().all()
        for m in rows:
            src = getattr(m, "sources", None)
            if not isinstance(src, dict):
                continue
            for key in ("images", "files"):
                for ref in (src.get(key) or []):
                    p = ref.get("path") if isinstance(ref, dict) else None
                    if p:
                        blob_paths.append(p)
    except Exception:  # noqa: BLE001 — never block the delete on collection
        pass

    ok = await SessionRepo(session).delete(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Drop the conversation's code knowledge graphs in the SAME transaction — a
    # single child-table DELETE, so it stays fast and atomic with the cascade.
    try:
        from sqlalchemy import text as _text

        await session.execute(
            _text("DELETE FROM code_graphs WHERE conversation_id = :cid"),
            {"cid": str(conversation_id)},
        )
    except Exception:  # noqa: BLE001
        pass
    await session.commit()

    # Owned-artifact cleanup (stored blobs on disk + RAG vectors in Qdrant) is
    # slow and NOT needed for the delete to be correct — run it detached so the
    # endpoint returns the instant the DB rows are committed. This is what made
    # deletion feel slow: the response used to block on disk + Qdrant I/O.
    _schedule_artifact_cleanup(str(conversation_id), blob_paths)

    return {"ok": True, "blobs_removed": len(blob_paths)}


@router.post("/conversations/{conversation_id}/cancel")
async def cancel_conversation_stream(conversation_id: str):
    """Stop an in-flight streaming turn for this conversation NOW.

    The FE's HTTP client can't abort a request mid-flight, so pressing Stop can't
    rely on the socket closing. This sets a cancel flag the streaming generators
    poll between steps — they then cancel the sandbox verify, close the LLM
    stream, save the partial, and end. Idempotent; safe to call when nothing is
    streaming."""
    from app.api.replay import request_cancel
    request_cancel(str(conversation_id))
    return {"ok": True}


@router.post("/conversations/bulk-delete")
async def bulk_delete_conversations(
    payload: dict,
    session: AsyncSession = Depends(get_session),
):
    """Delete many conversations at once. All rows (sessions + cascaded messages
    + code graphs) go in ONE transaction with a single commit, so removing 20
    chats costs one round-trip, not 20. Owned-artifact cleanup (blobs + vectors)
    is detached, exactly like the single delete."""
    from storage.repos import SessionRepo
    from sqlalchemy import text as _text

    ids = [str(i) for i in (payload.get("ids") or []) if i]
    if not ids:
        return {"ok": True, "deleted": 0, "blobs_removed": 0}

    # Collect owned blob paths across ALL targeted conversations up front.
    blob_paths: list[str] = []
    try:
        rows = (
            await session.execute(
                select(Message).where(Message.conversation_id.in_(ids))
            )
        ).scalars().all()
        for m in rows:
            src = getattr(m, "sources", None)
            if not isinstance(src, dict):
                continue
            for key in ("images", "files"):
                for ref in (src.get(key) or []):
                    p = ref.get("path") if isinstance(ref, dict) else None
                    if p:
                        blob_paths.append(p)
    except Exception:  # noqa: BLE001 — never block the delete on collection
        pass

    deleted = await SessionRepo(session).delete_many(ids)
    try:
        from sqlalchemy import bindparam as _bindparam

        await session.execute(
            _text("DELETE FROM code_graphs WHERE conversation_id IN :cids")
            .bindparams(_bindparam("cids", expanding=True)),
            {"cids": ids},
        )
    except Exception:  # noqa: BLE001
        pass
    await session.commit()

    # One detached cleanup task for the whole batch (blobs are already pooled;
    # drop each conversation's RAG collection).
    for cid in ids:
        _schedule_artifact_cleanup(cid, [])
    _schedule_artifact_cleanup("", blob_paths)  # blob files (no collection drop)

    return {"ok": True, "deleted": deleted, "blobs_removed": len(blob_paths)}
