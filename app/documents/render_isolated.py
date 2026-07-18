"""Subprocess-isolated rendering — Phase 1b of the Document Generation roadmap.

The Job Manager runs a render off the event loop; by default that render happens
IN-PROCESS (a worker thread). When ``cfg.documents.sandbox_render`` is on, the
deterministic render instead runs in a separate WORKER PROCESS so that:

  * a native crash in a renderer (a segfault in a C extension, an OOM in a huge
    export) takes the worker process, NOT the async server, and
  * a wedged render can be abandoned without corrupting the main interpreter.

The worker pool is a warm ``ProcessPoolExecutor`` (workers persist, so the
one-time import cost is amortized) initialized lazily and torn down at exit. The
worker entrypoint :func:`render_payload` is a TOP-LEVEL function taking only
picklable arguments (strings + the ``ExportSettings`` dataclass) — the roadmap's
"the sandbox receives a structured payload, never a raw prompt". Everything is
fail-open: a broken pool / timeout / pickling error is raised to the caller,
which falls back to an in-process render so a download never fails.
"""
from __future__ import annotations

import atexit
import concurrent.futures as _futures
from typing import Optional

# The worker imports are LAZY (inside render_payload) so spawning a worker pulls
# in app.documents.generators + the render libs only — not FastAPI / torch.

_POOL: Optional[_futures.ProcessPoolExecutor] = None


def render_payload(content: str, fmt: str, title: str = "",
                   export_settings=None, language: str = "") -> dict:
    """Run the full render→validate→repair→degrade loop and return a picklable
    result dict. TOP-LEVEL (not a closure) so a ``ProcessPoolExecutor`` can ship
    it to a worker process. Runs in the WORKER process when called via the pool;
    also callable in-process (it's a plain function) for testing."""
    try:
        from app.documents.validators import render_validated
        data, mime, ext, val_meta = render_validated(
            content, fmt, title=title, export_settings=export_settings,
            language=language)
        return {"data": data, "mime": mime, "ext": ext, "val_meta": val_meta}
    except Exception:  # noqa: BLE001 — validation guard failed → plain render
        from app.documents.generators import render_document
        data, mime, ext = render_document(
            content, fmt, title=title, export_settings=export_settings,
            language=language)
        return {"data": data, "mime": mime, "ext": ext, "val_meta": None}


def _max_workers() -> int:
    try:
        from app.core.config_loader import get_config
        return max(1, int(getattr(get_config().documents,
                                  "export_concurrency", 2) or 2))
    except Exception:  # noqa: BLE001
        return 2


def _pool() -> _futures.ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = _futures.ProcessPoolExecutor(max_workers=_max_workers())
    return _POOL


def render_isolated(content: str, fmt: str, title: str = "",
                    export_settings=None, timeout: Optional[float] = None,
                    language: str = "") -> dict:
    """Render in a worker PROCESS and return the payload dict. Raises on timeout
    (``concurrent.futures.TimeoutError``) or a crashed worker
    (``BrokenProcessPool`` — the pool is torn down + recreated so the NEXT call
    is healthy). Callers treat any exception as "fall back to in-process"."""
    try:
        fut = _pool().submit(render_payload, content, fmt, title,
                             export_settings, language)
        return fut.result(timeout=timeout)
    except _futures.process.BrokenProcessPool:
        # A worker died mid-render — poison the pool so it's rebuilt next time.
        shutdown_pool()
        raise


def shutdown_pool() -> None:
    """Tear the worker pool down (called at exit; also on a broken pool)."""
    global _POOL
    if _POOL is not None:
        pool, _POOL = _POOL, None
        try:
            pool.shutdown(cancel_futures=True)
        except TypeError:  # pragma: no cover — py<3.9 has no cancel_futures
            pool.shutdown()
        except Exception:  # noqa: BLE001
            pass


atexit.register(shutdown_pool)

__all__ = ["render_payload", "render_isolated", "shutdown_pool"]
