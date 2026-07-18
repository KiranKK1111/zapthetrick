"""Progressive context + incremental retrieval (perceived-speed R5, R6).

A single LLM call can't change its prompt mid-stream, so "incremental retrieval"
here means: launch retrieval CONCURRENTLY with the other pre-generation work
(routing, prompt assembly) instead of strictly before it, so retrieval latency
overlaps that work rather than adding to it (R6.1). When retrieval produces
usable snippets they are folded into the context exactly as up-front loading
would (so the final answer is equivalent — Property 6 / R5.3); when it yields
nothing the answer proceeds without grounding and is marked unavailable (R6.3).

Pure + fail-open: any retrieval error degrades to essential-only + unavailable.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class ContextResult:
    context: list                 # the assembled context (essential [+ retrieved])
    grounding: str                # "used" | "unavailable"


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class ProgressiveContext:
    """Overlaps retrieval with pre-generation prep, then folds usable results."""

    async def assemble(
        self,
        essential: list,
        retrieve: Callable[[], Any],
        inject: Callable[[list, list], list],
        *,
        prep: Callable[[], Any] | None = None,
    ) -> ContextResult:
        """Run `retrieve()` concurrently with `prep()` (if given), then fold the
        retrieved snippets into `essential` via `inject` when non-empty.

        - `retrieve` → awaitable/sync returning a list of snippets (or falsy).
        - `inject(essential, snippets)` → the combined context (must equal what
          up-front loading would build, to preserve answer equivalence).
        - `prep` → optional pre-generation work to overlap retrieval with.
        """
        ret_task = asyncio.ensure_future(_as_coro(retrieve))
        # Overlap: do the prep work while retrieval is in flight (R6.1).
        if prep is not None:
            try:
                await _maybe_await(prep())
            except Exception:  # noqa: BLE001 — prep failure shouldn't lose retrieval
                pass
        try:
            snippets = await ret_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — fail-open (R6.3)
            log.info("progressive retrieval failed (%s) — no grounding", exc)
            snippets = None
        if snippets:
            return ContextResult(inject(essential, list(snippets)), "used")
        return ContextResult(essential, "unavailable")


async def _as_coro(fn: Callable[[], Any]):
    """Call `fn()` and await it if needed — lets `retrieve` be sync or async."""
    return await _maybe_await(fn())


@dataclass
class ArtifactEvent:
    """One artifact becoming ready during progressive delivery (P5 #18)."""
    index: int          # completion order (0-based)
    name: str           # artifact label / id
    artifact: Any       # the produced artifact (or None on failure)
    ok: bool            # False when the producer raised
    error: str = ""


async def deliver_artifacts(
    producers: "dict[str, Callable[[], Any]] | list[tuple[str, Callable[[], Any]]]",
) -> "AsyncIterator[ArtifactEvent]":
    """Emit each artifact AS IT COMPLETES, not after the whole batch (P5 #18).

    `producers` maps a name → a sync/async callable that builds one artifact.
    They run concurrently; results are yielded in COMPLETION order (fastest
    first) so the UI can render each artifact the moment it's ready instead of
    waiting on the slowest. A producer that raises yields an `ok=False` event
    and never sinks its siblings. Fail-open: a framework error ends the stream
    cleanly rather than raising into the caller's render loop.
    """
    items = list(producers.items()) if isinstance(producers, dict) else list(producers)
    if not items:
        return
    tasks: dict[asyncio.Task, str] = {}
    try:
        for name, fn in items:
            tasks[asyncio.ensure_future(_as_coro(fn))] = name
    except Exception as exc:  # noqa: BLE001
        log.info("progressive artifact scheduling failed (%s)", exc)
        return
    index = 0
    pending = set(tasks)
    while pending:
        try:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            for t in pending:
                t.cancel()
            raise
        for task in done:
            name = tasks.get(task, "artifact")
            try:
                result = task.result()
                yield ArtifactEvent(index, name, result, True)
            except Exception as exc:  # noqa: BLE001 — one artifact failing is isolated
                yield ArtifactEvent(index, name, None, False, str(exc)[:200])
            index += 1


def progressive_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.perceived, "progressive_context", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["ProgressiveContext", "ContextResult", "progressive_enabled",
           "ArtifactEvent", "deliver_artifacts"]
