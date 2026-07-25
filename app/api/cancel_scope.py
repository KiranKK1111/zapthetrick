"""Per-turn cancellation scope (vNext §4.5).

STOP must halt EVERYTHING a turn spawned — not just break the step loop. Today
STOP is a cooperative flag (`app/api/replay`: request_cancel/is_cancelled) that
generators poll between steps. This adds the missing half: an ACTIVE teardown.

A per-turn ``CancellationScope`` (keyed by the turn id — conversation_id for
Chat; the Live qid registry composes) that every lane registers its cancellables
in: asyncio tasks/futures, sandbox process groups, and cleanup callbacks. On STOP
``scope.cancel()`` (a) sets the cooperative flag so existing pollers still stop,
then (b) actively cancels every registered lane. So a stopped turn leaves nothing
running — no background verify, no repair round, no post-answer critic.

Thread/loop-safe and fail-open: a teardown error never propagates, and a lane
registered AFTER the scope was cancelled is torn down immediately.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from app.api import replay as _replay

log = logging.getLogger(__name__)


def _safe_cancel_task(task: Any) -> None:
    try:
        if task is not None and not task.done():
            task.cancel()
    except Exception:  # noqa: BLE001
        pass


def _kill_sandbox(group_id: str) -> None:
    try:
        from app.sandbox import docker_exec as _dex
        _dex.cancel_group(group_id)
    except Exception:  # noqa: BLE001
        pass


def _safe_call(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


class CancellationScope:
    def __init__(self, turn_id: str):
        self.turn_id = turn_id
        self._lock = threading.RLock()
        self._tasks: list[Any] = []          # asyncio.Task/Future (.cancel())
        self._sandboxes: list[str] = []      # docker/bubblewrap process-group ids
        self._callbacks: list[Callable[[], Any]] = []
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        # Cancelled if this scope was cancelled OR the cooperative flag is set
        # (a STOP via the flag alone — e.g. a legacy path — still reads here).
        return self._cancelled or _replay.is_cancelled(self.turn_id)

    def register_task(self, task: Any) -> None:
        with self._lock:
            if not self._cancelled:
                self._tasks.append(task)
                return
        _safe_cancel_task(task)

    def register_sandbox(self, group_id: str) -> None:
        if not group_id:
            return
        with self._lock:
            if not self._cancelled:
                self._sandboxes.append(group_id)
                return
        _kill_sandbox(group_id)

    def register_callback(self, fn: Callable[[], Any]) -> None:
        with self._lock:
            if not self._cancelled:
                self._callbacks.append(fn)
                return
        _safe_call(fn)

    def cancel(self) -> None:
        with self._lock:
            already = self._cancelled
            self._cancelled = True
            tasks = self._tasks[:]
            sboxes = self._sandboxes[:]
            cbs = self._callbacks[:]
            self._tasks.clear()
            self._sandboxes.clear()
            self._callbacks.clear()
        # Cooperative flag first (cheap; unblocks step-polling generators).
        _replay.request_cancel(self.turn_id)
        if already:
            return
        for t in tasks:
            _safe_cancel_task(t)
        for g in sboxes:
            _kill_sandbox(g)
        for fn in cbs:
            _safe_call(fn)


# ── global registry ─────────────────────────────────────────────────────────
_LOCK = threading.RLock()
_SCOPES: dict[str, CancellationScope] = {}


def scope_for(turn_id: str) -> CancellationScope:
    """Get-or-create the scope for a turn. Lanes call
    ``scope_for(id).register_*`` as they spawn work."""
    with _LOCK:
        s = _SCOPES.get(turn_id)
        if s is None:
            s = _SCOPES[turn_id] = CancellationScope(turn_id)
        return s


def cancel(turn_id: str) -> None:
    """STOP: cancel a turn's scope (active teardown + cooperative flag). Safe to
    call when nothing is registered — the cooperative flag still fires so
    step-polling generators stop."""
    if not turn_id:
        return
    with _LOCK:
        s = _SCOPES.get(turn_id)
    if s is not None:
        s.cancel()
    else:
        _replay.request_cancel(turn_id)


def clear(turn_id: str) -> None:
    """Drop a finished (or freshly-starting) turn's scope + cooperative flag so a
    stale flag can't kill the next turn at birth."""
    if not turn_id:
        return
    with _LOCK:
        _SCOPES.pop(turn_id, None)
    _replay.clear_cancel(turn_id)


def reset_for_tests() -> None:
    with _LOCK:
        _SCOPES.clear()


__all__ = ["CancellationScope", "scope_for", "cancel", "clear"]
