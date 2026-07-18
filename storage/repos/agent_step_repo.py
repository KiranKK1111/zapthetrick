"""Agent-step repo — the ordered, replayable trace of a Code-In agent run.

One row per SSE event the agent loop emits, hanging off a `Session`
(`type="agent_code"`). Mirrors `MessageRepo`. The route holds a running `seq`
so we never do a `MAX(seq)` round-trip per event (this is hot — one append per
streamed event).
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select, update

from ..models import AgentStep
from .base import Repo


class AgentStepRepo(Repo):
    async def next_seq(self, session_id: uuid.UUID | str) -> tuple[int, int]:
        """`(next_seq, next_turn)` for resuming a session — one query at run
        start so the route can then increment `seq` in memory per event."""
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        result = await self.session.execute(
            select(func.max(AgentStep.seq), func.max(AgentStep.turn))
            .where(AgentStep.session_id == session_id)
        )
        mseq, mturn = result.one()
        return ((mseq if mseq is not None else -1) + 1,
                (mturn if mturn is not None else -1) + 1)

    async def append(
        self,
        *,
        session_id: uuid.UUID | str,
        seq: int,
        event: str,
        payload: dict,
        turn: int = 0,
        message_id: uuid.UUID | str | None = None,
        step: int | None = None,
        tool: str | None = None,
        kind: str | None = None,
        elapsed_ms: int | None = None,
        incomplete: bool = False,
    ) -> AgentStep:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        if isinstance(message_id, str):
            message_id = uuid.UUID(message_id)
        row = AgentStep(
            session_id=session_id,
            message_id=message_id,
            seq=seq,
            turn=turn,
            event=event,
            step=step,
            tool=tool,
            kind=kind,
            payload=payload or {},
            elapsed_ms=elapsed_ms,
            incomplete=incomplete,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_for_session(
        self, session_id: uuid.UUID | str, *, limit: int = 2000
    ) -> list[AgentStep]:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        result = await self.session.execute(
            select(AgentStep)
            .where(AgentStep.session_id == session_id)
            .order_by(AgentStep.seq)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_incomplete(self, session_id: uuid.UUID | str) -> None:
        """Flag the `final` step of a run as incomplete (interrupted run)."""
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        await self.session.execute(
            update(AgentStep)
            .where(AgentStep.session_id == session_id, AgentStep.event == "final")
            .values(incomplete=True)
        )
