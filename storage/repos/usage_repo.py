"""Per-call billing ledger — provider / model / token / latency rows."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import func, select

from ..models import ModelUsage
from .base import Repo


class UsageRepo(Repo):
    async def record(
        self,
        *,
        provider: str,
        model: str,
        role: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int | None = None,
        user_id: uuid.UUID | None = None,
    ) -> ModelUsage:
        row = ModelUsage(
            user_id=user_id,
            provider=provider,
            model=model,
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def daily_tokens(
        self, *, user_id: uuid.UUID | None = None, day: date | None = None
    ) -> int:
        """Total prompt + completion tokens for `day` (UTC). Drives any
        token-budget dashboards or hard caps."""
        target_day = day or datetime.now(timezone.utc).date()
        stmt = select(
            func.coalesce(
                func.sum(ModelUsage.prompt_tokens + ModelUsage.completion_tokens),
                0,
            )
        ).where(func.date(ModelUsage.occurred_at) == target_day)
        if user_id is not None:
            stmt = stmt.where(ModelUsage.user_id == user_id)
        scalar = await self.session.execute(stmt)
        return int(scalar.scalar_one() or 0)
