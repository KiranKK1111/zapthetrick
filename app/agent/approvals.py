"""Interactive tool approval for `ask` mode.

When the agent wants a write/bash in `ask` mode, the loop creates an approval id,
streams an `approval` event to the UI, and AWAITS this future. The UI answers via
`POST /api/agent/approve {id, allow}`, which resolves it. A timeout denies, so an
abandoned stream can't hang the loop forever.
"""
from __future__ import annotations

import asyncio
import secrets

_pending: dict[str, asyncio.Future] = {}


def create() -> str:
    aid = secrets.token_hex(6)
    _pending[aid] = asyncio.get_running_loop().create_future()
    return aid


async def wait(aid: str, *, timeout: float = 300.0) -> bool:
    fut = _pending.get(aid)
    if fut is None:
        return False
    try:
        return bool(await asyncio.wait_for(fut, timeout=timeout))
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        return False
    finally:
        _pending.pop(aid, None)


def resolve(aid: str, allow: bool) -> bool:
    fut = _pending.get(aid)
    if fut is not None and not fut.done():
        fut.set_result(bool(allow))
        return True
    return False


__all__ = ["create", "wait", "resolve"]
