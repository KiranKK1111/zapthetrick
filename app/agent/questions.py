"""Interactive USER questions + plan approval for the agent loop (Claude-style).

The agent calls `ask_user` (a clarifying question with options) or `present_plan`
(plan → approve → execute); the loop creates an id, streams a `question`/`plan`
event, and AWAITS this future. The UI answers via `POST /api/agent/answer
{id, answer}`, which resolves it. A timeout returns None so an abandoned stream
can't hang the loop forever.

Mirrors `approvals.py`, but the answer is a STRING (the chosen option / plan
decision / free text) rather than a boolean.
"""
from __future__ import annotations

import asyncio
import secrets

_pending: dict[str, asyncio.Future] = {}


def create() -> str:
    qid = secrets.token_hex(6)
    _pending[qid] = asyncio.get_running_loop().create_future()
    return qid


async def wait(qid: str, *, timeout: float = 600.0) -> str | None:
    fut = _pending.get(qid)
    if fut is None:
        return None
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        return None
    finally:
        _pending.pop(qid, None)


def resolve(qid: str, answer: str) -> bool:
    fut = _pending.get(qid)
    if fut is not None and not fut.done():
        fut.set_result(str(answer))
        return True
    return False


__all__ = ["create", "wait", "resolve"]
