"""Document Job Manager — Phase 1b of the Document Generation roadmap.

DocuementGeneration.md's addition recommends a **Sandbox Job Manager** rather
than invoking rendering directly, providing queue, priority, retry, cancellation,
progress events, timeout handling, resource limits, and cleanup. This is that
execution engine.

Relationship to the existing pieces:
  * `app/obs/jobs.py::JobRegistry` is the Task Center's STATUS BOARD (start /
    update / finish for the UI). It does not execute anything — this manager
    does, and reports lifecycle to it so document jobs show up there.
  * Rendering runs off the event loop (via ``asyncio.to_thread``) so heavy
    exports (openpyxl / python-docx / PyMuPDF) don't stall the async server and
    several can run concurrently under a bounded worker pool. The render
    function is INJECTABLE, so a true sandbox-subprocess executor (for crash
    isolation + hard kill on cancel) can be swapped in later without touching the
    manager — the current thread executor abandons an in-flight render on cancel
    (the thread finishes in the background; its result is discarded).

In-memory, single-process, fail-open — matching the rest of the app.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


_TERMINAL = {JobStatus.DONE, JobStatus.FAILED,
             JobStatus.CANCELLED, JobStatus.TIMEOUT}


@dataclass
class RenderJob:
    id: str
    content: str
    fmt: str
    title: str = ""
    priority: int = 0
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    attempts: int = 0
    result: Optional[bytes] = None
    mime: str = ""
    ext: str = ""
    error: str = ""
    created_at: float = 0.0
    finished_at: float = 0.0
    render_fn: Optional["RenderFn"] = None  # per-job override of the manager's

    @property
    def done(self) -> bool:
        return self.status in _TERMINAL

    def summary(self) -> dict:
        return {
            "id": self.id, "fmt": self.fmt, "status": self.status.value,
            "progress": round(self.progress, 3), "stage": self.stage,
            "attempts": self.attempts, "error": self.error,
            "bytes": len(self.result) if self.result else 0,
        }


# (content, fmt, title) -> (bytes, media_type, ext)
RenderFn = Callable[[str, str, str], tuple[bytes, str, str]]
ProgressCb = Callable[[RenderJob], None]


def _default_render(content: str, fmt: str, title: str) -> tuple[bytes, str, str]:
    from app.documents.generators import render_document
    return render_document(content, fmt, title)


# Deterministic render failures: the render function is PURE over (content, fmt,
# title), so these fail identically on every attempt — retrying just burns a
# worker and delays the user's error. Everything else (I/O, a wedged renderer
# lib, a transient MemoryError/OSError, a flaky sandbox) is treated as transient
# and IS retried.
_PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    TypeError, AttributeError, ImportError, NotImplementedError,
    KeyError, IndexError, SyntaxError, RecursionError, UnicodeDecodeError,
)


def _is_permanent(exc: BaseException) -> bool:
    """True when re-running the render can only fail the same way (unsupported
    format / malformed input / missing renderer / programming error)."""
    try:                            # an unsupported format never becomes supported
        from app.documents.generators import UnsupportedFormat
        if isinstance(exc, UnsupportedFormat):
            return True
    except Exception:  # noqa: BLE001 — classification must never raise
        pass
    return isinstance(exc, _PERMANENT_ERRORS)


class DocumentJobManager:
    """A bounded worker pool that renders documents as cancellable, prioritized,
    retryable jobs with progress + timeout."""

    def __init__(
        self, *,
        max_concurrent: int = 2,
        timeout_s: float = 120.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.2,
        render_fn: Optional[RenderFn] = None,
        clock: Callable[[], float] = time.monotonic,
        track: bool = True,
    ) -> None:
        self._max = max(1, int(max_concurrent))
        self._timeout = float(timeout_s)
        self._max_retries = max(0, int(max_retries))
        self._backoff = max(0.0, float(retry_backoff_s))
        self._render = render_fn or _default_render
        self._clock = clock
        self._track = track
        self._jobs: dict[str, RenderJob] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._on_progress: dict[str, ProgressCb] = {}
        self._track_id: dict[str, str] = {}
        # Queue + workers are bound to the event loop they're started on, so we
        # create them in start() (not here) and recreate on a loop change.
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._workers: list[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._seq = 0
        self._started = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        alive = self._started and self._loop is loop and any(
            not w.done() for w in self._workers)
        if alive:
            return
        # First start, or the loop changed (e.g. a new test event loop): (re)bind
        # the queue + worker pool to the CURRENT loop.
        self._queue = asyncio.PriorityQueue()
        self._loop = loop
        self._started = True
        self._workers = [asyncio.ensure_future(self._worker())
                         for _ in range(self._max)]

    async def aclose(self) -> None:
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await w
        self._workers = []
        self._started = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # ── submit / query ──────────────────────────────────────────────────────
    def submit(self, content: str, fmt: str, title: str = "", *,
               priority: int = 0, job_id: Optional[str] = None,
               on_progress: Optional[ProgressCb] = None,
               render_fn: Optional[RenderFn] = None) -> str:
        if self._queue is None:
            raise RuntimeError("DocumentJobManager.start() must run before submit")
        self._seq += 1
        jid = job_id or f"doc-job-{self._seq}"
        job = RenderJob(id=jid, content=content, fmt=fmt, title=title,
                        priority=int(priority), created_at=self._clock(),
                        render_fn=render_fn)
        self._jobs[jid] = job
        self._done[jid] = asyncio.Event()
        self._cancels[jid] = asyncio.Event()
        if on_progress is not None:
            self._on_progress[jid] = on_progress
        if self._track:
            with contextlib.suppress(Exception):
                from app.obs.jobs import jobs as _reg
                self._track_id[jid] = _reg().start(
                    f"Export · {(fmt or 'document').upper()}", kind="export")
        # Higher priority first; FIFO within a priority via the sequence.
        self._queue.put_nowait((-job.priority, self._seq, jid))
        return jid

    def get(self, job_id: str) -> Optional[RenderJob]:
        return self._jobs.get(job_id)

    async def wait(self, job_id: str) -> RenderJob:
        ev = self._done.get(job_id)
        if ev is not None:
            await ev.wait()
        return self._jobs[job_id]

    async def submit_and_wait(self, content: str, fmt: str, title: str = "",
                              **kw) -> RenderJob:
        jid = self.submit(content, fmt, title, **kw)
        return await self.wait(jid)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. A QUEUED job never runs; a RUNNING job abandons
        its in-flight render (see the thread-executor caveat in the module doc).
        Returns False if unknown or already finished."""
        job = self._jobs.get(job_id)
        ev = self._cancels.get(job_id)
        if job is None or ev is None or job.done:
            return False
        ev.set()
        return True

    def cleanup(self, ttl_s: float = 3600.0) -> int:
        """Drop finished jobs older than ``ttl_s``. Returns the number removed."""
        now = self._clock()
        stale = [jid for jid, j in self._jobs.items()
                 if j.done and j.finished_at and (now - j.finished_at) > ttl_s]
        for jid in stale:
            self._jobs.pop(jid, None)
            self._done.pop(jid, None)
            self._cancels.pop(jid, None)
            self._on_progress.pop(jid, None)
            self._track_id.pop(jid, None)
        return len(stale)

    def stats(self) -> dict:
        by = {s.value: 0 for s in JobStatus}
        for j in self._jobs.values():
            by[j.status.value] += 1
        return {"total": len(self._jobs), "by_status": by,
                "workers": len(self._workers), "queued": self._queue.qsize()}

    # ── execution ───────────────────────────────────────────────────────────
    async def _worker(self) -> None:
        while True:
            _, _, jid = await self._queue.get()
            try:
                job = self._jobs.get(jid)
                if job is not None:
                    await self._execute(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a worker must not die
                job = self._jobs.get(jid)
                if job is not None and not job.done:
                    self._finish(job, JobStatus.FAILED, error=str(exc)[:200])
            finally:
                self._queue.task_done()

    async def _execute(self, job: RenderJob) -> None:
        """Run the job with bounded retry: each attempt gets the FULL timeout, a
        transient failure (or timeout) is retried after an exponential backoff,
        a DETERMINISTIC failure fails fast (see :func:`_is_permanent`), and a
        cancellation — before, during, or between attempts — always wins."""
        cancel = self._cancels[job.id]
        for attempt in range(self._max_retries + 1):
            if cancel.is_set():                    # never (re)run a cancelled job
                self._finish(job, JobStatus.CANCELLED, stage="cancelled")
                return
            last = attempt >= self._max_retries
            job.attempts = attempt + 1
            self._transition(job, JobStatus.RUNNING, 0.1,
                             "rendering" if attempt == 0
                             else f"rendering (retry {attempt})")
            _render = job.render_fn or self._render
            render_task = asyncio.ensure_future(
                asyncio.to_thread(_render, job.content, job.fmt, job.title))
            cancel_task = asyncio.ensure_future(cancel.wait())
            done, _pending = await asyncio.wait(
                {render_task, cancel_task}, timeout=self._timeout,
                return_when=asyncio.FIRST_COMPLETED)

            if cancel_task in done:
                await _kill(render_task)
                self._finish(job, JobStatus.CANCELLED, stage="cancelled")
                return
            await _kill(cancel_task)

            if render_task not in done:            # timed out
                await _kill(render_task)
                if last:
                    self._finish(job, JobStatus.TIMEOUT, error="render timed out")
                    return
                if not await self._backoff_wait(job, cancel, attempt):
                    return                          # cancelled while backing off
                continue
            try:
                data, mime, ext = render_task.result()
            except Exception as exc:  # noqa: BLE001
                if last or _is_permanent(exc):
                    self._finish(job, JobStatus.FAILED, error=str(exc)[:200])
                    return
                if not await self._backoff_wait(job, cancel, attempt):
                    return                          # cancelled while backing off
                continue
            job.result, job.mime, job.ext = data, mime, ext
            self._finish(job, JobStatus.DONE, stage="done")
            return

    async def _backoff_wait(self, job: RenderJob, cancel: asyncio.Event,
                            attempt: int) -> bool:
        """Sleep the exponential backoff before the next attempt, staying
        cancellable. Returns False when the job was cancelled while waiting (it
        is already finished in that case)."""
        delay = self._backoff * (2 ** attempt)
        if delay > 0:
            self._transition(job, JobStatus.RUNNING, job.progress,
                             f"retrying in {delay:.2f}s")
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(cancel.wait(), timeout=delay)
        if cancel.is_set():
            self._finish(job, JobStatus.CANCELLED, stage="cancelled")
            return False
        return True

    # ── state transitions + reporting ───────────────────────────────────────
    def _transition(self, job: RenderJob, status: JobStatus,
                    progress: float, stage: str) -> None:
        job.status = status
        job.progress = progress
        job.stage = stage
        self._report(job)

    def _finish(self, job: RenderJob, status: JobStatus, *,
                stage: str = "", error: str = "") -> None:
        job.status = status
        job.stage = stage or status.value
        job.progress = 1.0 if status == JobStatus.DONE else job.progress
        job.error = error
        job.finished_at = self._clock()
        self._report(job)
        ev = self._done.get(job.id)
        if ev is not None:
            ev.set()

    def _report(self, job: RenderJob) -> None:
        cb = self._on_progress.get(job.id)
        if cb is not None:
            with contextlib.suppress(Exception):
                cb(job)
        if self._track:
            tid = self._track_id.get(job.id)
            if tid:
                with contextlib.suppress(Exception):
                    from app.obs.jobs import jobs as _reg
                    if job.done:
                        _reg().finish(tid, ok=(job.status == JobStatus.DONE),
                                      detail=job.error[:80] or None)
                    else:
                        _reg().update(tid, progress=job.progress,
                                      detail=job.stage)


async def _kill(task: asyncio.Task) -> None:
    """Cancel and drain a task, swallowing whatever it raises."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ── process-wide singleton (the live export path submits here) ──────────────
_MANAGER: Optional[DocumentJobManager] = None


async def get_manager() -> DocumentJobManager:
    """The shared document job manager, workers started (idempotent). Concurrency
    + timeout come from cfg.documents when present, else safe defaults. Tracking
    is OFF here — callers that already register a Task Center job (the export
    endpoint) pass their own; a bare submit still shows via the caller."""
    global _MANAGER
    loop = asyncio.get_running_loop()
    # Recreate on a loop change so queue/worker state never dangles on a dead
    # loop (safe no-op in production — one loop for the app's lifetime).
    if _MANAGER is None or getattr(_MANAGER, "_loop", None) not in (None, loop):
        conc, timeout = 2, 120.0
        retries, backoff = 1, 0.2
        try:
            from app.core.config_loader import get_config
            _d = get_config().documents
            conc = int(getattr(_d, "export_concurrency", conc) or conc)
            timeout = float(getattr(_d, "export_timeout_s", timeout) or timeout)
            # `0` is a legitimate value (retry off) → don't `or` a default over it.
            _r = getattr(_d, "export_max_retries", None)
            if _r is not None:
                retries = max(0, int(_r))
            _b = getattr(_d, "export_retry_backoff_s", None)
            if _b is not None:
                backoff = max(0.0, float(_b))
        except Exception:  # noqa: BLE001
            pass
        _MANAGER = DocumentJobManager(
            max_concurrent=conc, timeout_s=timeout, max_retries=retries,
            retry_backoff_s=backoff, track=False)
    await _MANAGER.start()
    return _MANAGER


__all__ = ["JobStatus", "RenderJob", "DocumentJobManager", "get_manager"]
