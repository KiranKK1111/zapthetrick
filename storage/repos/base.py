"""Common base — one constructor signature for every repo.

Every repo takes an [AsyncSession] in its constructor and holds it
read-only. Multiple repos can share a session inside one HTTP request
(they all see the same transaction).

The dependency wrapper [get_session] in [app.storage.db] yields the
session; route handlers compose repos like:

    async def some_route(session = Depends(get_session)):
        sessions = SessionRepo(session)
        msgs    = MessageRepo(session)
        ...
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


class Repo:
    """Common base — one constructor signature, no behaviour."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
