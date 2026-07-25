"""Warm sandbox pool (vNext §3.1) — pre-created, ready-to-use sandbox
workspaces so a verify run pays ~20-50 ms setup instead of a cold
`mkdtemp` + runtime cold-start each time. This is what makes "verify every
code answer" free-feeling (§3.1).

**The isolation guarantee is unchanged.** A workspace handed out by
`acquire()` is used for exactly ONE run and then DESTROYED by `release()` —
never reused. The pool amortizes *creation* (and optional runtime page-cache
warmth), never the workspace itself, so the sandbox's "throw the world away"
contract holds byte-for-byte.

**Fail-open.** When the pool is off or a pool op errors, callers fall straight
through to a cold `tempfile.mkdtemp` — identical to before, so a pool hiccup
can never break a run.

**Where it helps.** No-op win on the `docker` backend (the sandbox container is
already persistent — `docker exec` has no cold start); the payoff is the
`local`/namespace (bwrap) path where each run otherwise pays workspace +
runtime cold start. It is also the seam Stage-4 Components B
(verify-while-streaming) and H (toolchain prefetch) hook into via
`warm_toolchain()`.

Thread-based (not asyncio) because the executor is synchronous subprocess code,
typically driven from `asyncio.to_thread`; a daemon refiller keeps the pool
topped up off the request path.
"""
from __future__ import annotations

import atexit
import logging
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

log = logging.getLogger(__name__)


class SandboxPool:
    """A bounded pool of pre-created, empty workspace directories.

    `acquire()` never blocks a run: it returns a ready workspace if one is
    waiting, else creates a fresh one on the spot. `release()` DESTROYS the
    workspace (rmtree) — it is never returned to the pool. A background daemon
    keeps ~`size` workspaces ready.
    """

    def __init__(self, size: int = 3) -> None:
        self._size = max(0, int(size))
        self._ready: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._created = 0      # total workspaces ever created (tests/telemetry)
        self._destroyed = 0    # total workspaces ever destroyed
        for _ in range(self._size):
            try:
                self._ready.put_nowait(self._create())
            except Exception:  # noqa: BLE001 — a failed prefill just means colder
                break
        self._refiller = threading.Thread(
            target=self._refill_loop, name="sandbox-pool-refill", daemon=True)
        self._refiller.start()
        atexit.register(self.close)

    # -- lifecycle ------------------------------------------------------------
    def _create(self) -> str:
        ws = tempfile.mkdtemp(prefix="dtt-sbx-pool-")
        with self._lock:
            self._created += 1
        return ws

    def acquire(self) -> str:
        """A ready workspace (or a fresh one on a pool miss). Never blocks."""
        try:
            return self._ready.get_nowait()
        except queue.Empty:
            return self._create()

    def release(self, ws: str) -> None:
        """Destroy `ws` — the throw-the-world-away guarantee. Never reused."""
        try:
            shutil.rmtree(ws, ignore_errors=True)
        finally:
            with self._lock:
                self._destroyed += 1

    def _refill_loop(self) -> None:
        while not self._closed:
            try:
                if self._ready.qsize() < self._size:
                    self._ready.put_nowait(self._create())
                    continue
            except Exception:  # noqa: BLE001 — refill is best-effort
                pass
            time.sleep(0.05)

    def close(self) -> None:
        """Stop refilling and destroy every ready workspace (idempotent)."""
        self._closed = True
        while True:
            try:
                ws = self._ready.get_nowait()
            except queue.Empty:
                break
            shutil.rmtree(ws, ignore_errors=True)

    # -- introspection --------------------------------------------------------
    def ready(self) -> int:
        return self._ready.qsize()

    def stats(self) -> dict:
        with self._lock:
            return {"size": self._size, "ready": self._ready.qsize(),
                    "created": self._created, "destroyed": self._destroyed}


# ---- module singleton -------------------------------------------------------
_pool: SandboxPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> SandboxPool:
    """The process-wide pool, sized from `cfg.sandbox.pool_size` (lazy)."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            size = 3
            try:
                from app.core.config_loader import cfg
                size = int(getattr(cfg.sandbox, "pool_size", 3))
            except Exception:  # noqa: BLE001
                size = 3
            _pool = SandboxPool(size=size)
    return _pool


def pool_enabled() -> bool:
    """Whether the warm pool is switched on (default OFF → cold mkdtemp)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.sandbox, "warm_pool", False))
    except Exception:  # noqa: BLE001
        return False


def reset_pool() -> None:
    """Tear down the singleton (tests / config reload)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


# ---- toolchain warmth (best-effort, for Components B & H) -------------------
# A trivial run pages the interpreter/runtime into the OS cache so the next real
# run starts hot. Best-effort and non-fatal: warmth is a bonus, never required.
_WARM_PROBES: dict[str, list[str]] = {
    "python": [sys.executable or "python3", "-I", "-B", "-c", "pass"],
    "node": ["node", "-e", "0"],
    "javascript": ["node", "-e", "0"],
    "ruby": ["ruby", "-e", ""],
    "php": ["php", "-r", ";"],
}
_warmed: set[str] = set()
_warm_lock = threading.Lock()


def warm_toolchain(language: str) -> None:
    """Page in `language`'s runtime once (idempotent, swallow-all). Called at
    language detection by Stage-4 §3.5 so the verifier starts hot."""
    lang = (language or "").strip().lower()
    probe = _WARM_PROBES.get(lang)
    if not probe:
        return
    with _warm_lock:
        if lang in _warmed:
            return
        _warmed.add(lang)
    try:
        subprocess.run(probe, capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 — warmth is never load-bearing
        pass


def prefetch_enabled() -> bool:
    """Stage-4 §3.5 toolchain prefetch — fire `prefetch_toolchain` from the
    streaming path the moment a code answer's language is known, so the verifier
    starts on a hot runtime. Default OFF → today's cold verify start."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.code_solver, "toolchain_prefetch", False))
    except Exception:  # noqa: BLE001
        return False


import re as _re

_OPEN_FENCE = _re.compile(r"(?:^|\n)```([\w+#.\-]+)")


def first_fence_language(text: str) -> str | None:
    """The language tag of the FIRST opening code fence in `text` (possibly a
    partial mid-stream buffer), lowercased — or None. Used to warm the toolchain
    as soon as a fenced block starts, before the answer finishes streaming."""
    try:
        m = _OPEN_FENCE.search(text or "")
        return m.group(1).strip().lower() if m else None
    except Exception:  # noqa: BLE001
        return None


def prefetch_toolchain(language: str) -> None:
    """Fire-and-forget `warm_toolchain` on a daemon thread (§3.5) — the moment a
    chat answer's language resolves, warm the runtime OFF the request path so the
    verifier starts hot. Idempotent + swallow-all; no-op for un-probed languages."""
    lang = (language or "").strip().lower()
    if not lang or lang not in _WARM_PROBES:
        return
    with _warm_lock:
        if lang in _warmed:
            return
    try:
        threading.Thread(target=warm_toolchain, args=(lang,),
                         name=f"warm-{lang}", daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["SandboxPool", "get_pool", "pool_enabled", "reset_pool",
           "warm_toolchain", "prefetch_toolchain", "prefetch_enabled",
           "first_fence_language"]
