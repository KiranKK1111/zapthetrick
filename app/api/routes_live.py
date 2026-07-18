"""
Live-interview session persistence.

The Live module stores each interview as a `Session(type="live")` titled with
the organization conducting it, plus the detected questions and the assistant's
answers as `Message` rows — exactly like a chat conversation, so the same
history/sidebar machinery applies. The WebSocket (`routes_ws.py`) writes the
Q&A rows; this router creates and lists the sessions.

Loading one session's transcript and deleting it reuse the existing
`GET/DELETE /api/conversations/{id}` endpoints (a live session IS a session).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session

router = APIRouter(prefix="/api/live")


class CreateLiveSession(BaseModel):
    org_name: str = ""
    job_role: str = ""
    job_description: str = ""
    notes: str = ""
    # Seniority override for answer calibration: "auto" (default) or a band slug
    # (intern / fresher / junior / mid / senior / lead / principal / distinguished).
    experience_level: str = ""


def _live_summary(row) -> dict:
    md = row.session_metadata or {}
    return {
        "id": str(row.id),
        "title": row.title or "Interview",
        "org_name": md.get("org_name", row.title or ""),
        "job_role": md.get("job_role", ""),
        "job_description": md.get("job_description", ""),
        "notes": md.get("notes", ""),
        "experience_level": md.get("experience_level", ""),
        "resume_id": str(row.resume_id) if getattr(row, "resume_id", None) else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "message_count": int(getattr(row, "message_count", 0) or 0),
        "last_message_at": (
            row.last_message_at.isoformat()
            if getattr(row, "last_message_at", None) else None
        ),
    }


@router.post("/sessions")
async def create_live_session(
    body: CreateLiveSession,
    session: AsyncSession = Depends(get_session),
):
    """Create a live-interview session titled with the organization name.
    The returned `id` is passed to the WebSocket as `?session_id=` so the
    Q&A gets persisted under it."""
    from storage.repos import SessionRepo

    org = (body.org_name or "").strip() or "Interview"
    repo = SessionRepo(session)
    row = await repo.create(
        type="live",
        title=org,
        session_metadata={
            "org_name": org,
            "job_role": (body.job_role or "").strip(),
            "job_description": (body.job_description or "").strip(),
            "notes": (body.notes or "").strip(),
            "experience_level": (body.experience_level or "").strip().lower(),
        },
    )
    await session.commit()
    return _live_summary(row)


@router.get("/ledger/summary")
async def ledger_summary():
    """Accuracy-ledger counters since startup: decisions by kind/reason plus
    user feedback tallies. The full labeled log is the JSONL on disk."""
    from app.live import ledger

    return ledger.summary()


@router.get("/sessions")
async def list_live_sessions(
    limit: int = 200,
    session: AsyncSession = Depends(get_session),
):
    """List past live-interview sessions (org name + created date) for the
    Live sidebar, most recent first."""
    from storage.repos import SessionRepo

    try:
        rows = await SessionRepo(session).list(type="live", limit=limit)
        return [_live_summary(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"list_live_sessions: {exc}")
