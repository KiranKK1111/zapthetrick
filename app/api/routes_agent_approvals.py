"""Agent approval/question endpoints — answer an `ask`-mode approval or an
`ask_user` question the shared workspace tool loop is awaiting.

Used by the CHAT agent-run path (/api/chat/agent-run streams `approval` and
`question` events; the FE timeline resolves them here). Extracted from the
removed Code-In module (2026-07-08) because chat reuses these two endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/agent", tags=["agent"])


class ApproveBody(BaseModel):
    id: str
    allow: bool


@router.post("/approve")
async def approve(body: ApproveBody) -> dict:
    """Answer an `ask`-mode approval the loop is awaiting."""
    from app.agent import approvals
    ok = approvals.resolve(body.id, body.allow)
    return {"resolved": ok}


class AnswerBody(BaseModel):
    id: str
    answer: str


@router.post("/answer")
async def answer(body: AnswerBody) -> dict:
    """Answer an `ask_user` question / `present_plan` decision the loop awaits."""
    from app.agent import questions
    ok = questions.resolve(body.id, body.answer)
    return {"resolved": ok}
