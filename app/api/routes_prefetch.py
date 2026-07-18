"""Prefetch endpoint (perceived-speed R1).

`POST /api/prefetch` is called (debounced) while the user is typing. It predicts
the request shape LOCALLY and warms the path (pooled connection + handles),
returning a `prefetch_token` the client echoes back on submit so the warmed work
is reused. It NEVER starts answer generation, and it is a cheap no-op when
`cfg.perceived.speculation_enabled` is False (returns `{"token": null}`).
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.perceived.prefetch import manager

router = APIRouter(prefix="/api")


class PrefetchBody(BaseModel):
    partial: str = ""


@router.post("/prefetch")
async def prefetch(body: PrefetchBody) -> dict:
    """Warm while typing. Returns a token (or null when speculation is off)."""
    token = await manager.warm(body.partial or "")
    return {"token": token}
