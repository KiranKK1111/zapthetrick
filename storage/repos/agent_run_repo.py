"""Agent-run ledger — per-tick agent execution metrics.

The supervisor opens a row when an agent starts and closes it when the
agent finishes / times out / errors. Used by the trace view and any
cost analytics.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from ..models import AgentRun
from .base import Repo


class AgentRunRepo(Repo):
    async def start(
        self,
        *,
        agent: str,
        session_id: uuid.UUID | str | None = None,
        message_id: uuid.UUID | str | None = None,
        input_summary: dict | None = None,
    ) -> AgentRun:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        if isinstance(message_id, str):
            message_id = uuid.UUID(message_id)
        row = AgentRun(
            agent=agent,
            session_id=session_id,
            message_id=message_id,
            input_summary=input_summary,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def finish(
        self,
        run_id: uuid.UUID | str,
        *,
        status: str,
        output_summary: dict | None = None,
        tokens: int | None = None,
        cost_estimate: float | None = None,
    ) -> None:
        if isinstance(run_id, str):
            run_id = uuid.UUID(run_id)
        row = await self.session.get(AgentRun, run_id)
        if row is None:
            return
        row.status = status
        row.ended_at = datetime.now(timezone.utc)
        row.output_summary = output_summary
        row.tokens = tokens
        row.cost_estimate = cost_estimate

    async def list_for_session(
        self, session_id: uuid.UUID | str, *, limit: int = 200
    ) -> list[AgentRun]:
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)
        result = await self.session.execute(
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .order_by(AgentRun.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
