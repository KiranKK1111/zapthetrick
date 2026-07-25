"""GPU admission & scheduling plane (vNext §9.1, Stage 6 Component L).

One GPU serves many jobs with very different urgency: STT/VAD must run within
milliseconds, the VLM and reranker are interactive (deadline-bound, sheddable to
cloud), and rasters / pre-answers / eval are background (idle-only). Without a
single admission controller they contend blindly — a background raster can evict
the VRAM a live answer needs mid-interview. §9.1 adds that controller: three
LANES over a VRAM reservation LEDGER, a `max_live_sessions` cap, and a defined
DEGRADE ORDER so contention sheds the least-important work first.

  * **realtime** (STT/VAD) — never rejected, ≤5 ms; reserves best-effort even
    under pressure (dropping speech is not acceptable).
  * **interactive** (VLM / T4 / reranker) — admitted if VRAM fits, else SHED to
    the cloud path rather than queueing behind a raster.
  * **background** (rasters / pre-answers / screen / eval) — idle-only: admitted
    only when no interactive work is active AND free VRAM leaves a headroom
    margin; otherwise deferred.

The real VRAM signal + the vLLM 4-bit VLM (§1.1) are pod-side; this module is the
pure admission LOGIC (injectable VRAM total + clock) so it is unit-tested on any
box and the real reservation activates on the pod. Fail-open: any error admits
(never block a real job on a scheduler bug). Flag-gated (`gpu.scheduler`, OFF).
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

# Lanes.
REALTIME = "realtime"
INTERACTIVE = "interactive"
BACKGROUND = "background"

# Admission actions.
RUN = "run"                 # admitted; VRAM reserved
SHED_CLOUD = "shed_cloud"   # interactive can't fit locally → cloud path
DEFER = "defer"             # background deferred (busy / no headroom)
REJECT = "reject"           # a new Live session over the cap

# §9.1 degrade order — the work contention sheds FIRST → last. A pre-answer is the
# cheapest to lose; the VLM is shed only as a last resort.
DEGRADE_ORDER = ("pre_answer", "screen", "speculation_top1", "vlm")


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.gpu, "scheduler", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Admission:
    admit: bool
    action: str            # run | shed_cloud | defer | reject
    reason: str = ""


@dataclass
class GpuScheduler:
    """Single GPU admission controller. Deterministic, in-process, fail-open."""
    total_vram_mb: int = 24_000
    max_live_sessions: int = 2
    bg_headroom_mb: int = 1_000
    now: Callable[[], float] | None = None

    _holders: dict = field(default_factory=dict)   # holder -> (lane, mb)
    _sessions: set = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── VRAM ledger ────────────────────────────────────────────────────────
    @property
    def reserved_mb(self) -> int:
        with self._lock:
            return sum(mb for _lane, mb in self._holders.values())

    @property
    def free_mb(self) -> int:
        return max(0, self.total_vram_mb - self.reserved_mb)

    def _active(self, lane: str) -> int:
        with self._lock:
            return sum(1 for ln, _mb in self._holders.values() if ln == lane)

    # ── sessions (multi-user contract) ─────────────────────────────────────
    def open_session(self, session_id: str) -> bool:
        """Admit a new Live session unless the `max_live_sessions` cap is hit.
        Idempotent for an already-open session."""
        with self._lock:
            if session_id in self._sessions:
                return True
            if len(self._sessions) >= max(1, self.max_live_sessions):
                return False
            self._sessions.add(session_id)
            return True

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.discard(session_id)

    @property
    def live_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ── admission ──────────────────────────────────────────────────────────
    def admit(self, lane: str, holder: str, *, est_mb: int = 0) -> Admission:
        """Decide whether `holder` may run on `lane` needing `est_mb` VRAM. On a
        RUN the VRAM is reserved (release it with `release(holder)`). Never
        raises → admits on error (a scheduler bug must not stall a real job)."""
        try:
            if lane == REALTIME:
                # Realtime never waits and is never rejected — reserve regardless.
                self._reserve(holder, REALTIME, est_mb)
                return Admission(True, RUN, "realtime — always admitted")
            if lane == INTERACTIVE:
                if est_mb <= self.free_mb:
                    self._reserve(holder, INTERACTIVE, est_mb)
                    return Admission(True, RUN, "fits VRAM")
                return Admission(False, SHED_CLOUD,
                                 "insufficient VRAM — shed to cloud")
            if lane == BACKGROUND:
                if self._active(INTERACTIVE) > 0 or self._active(REALTIME) > 0:
                    return Admission(False, DEFER, "GPU busy — idle-only lane")
                if est_mb + self.bg_headroom_mb <= self.free_mb:
                    self._reserve(holder, BACKGROUND, est_mb)
                    return Admission(True, RUN, "idle + headroom")
                return Admission(False, DEFER, "no headroom for background")
            # Unknown lane → admit (fail-open).
            return Admission(True, RUN, f"unknown lane {lane!r} — admitted")
        except Exception:  # noqa: BLE001
            return Admission(True, RUN, "scheduler error — fail-open admit")

    def _reserve(self, holder: str, lane: str, mb: int) -> None:
        with self._lock:
            self._holders[holder] = (lane, max(0, int(mb)))

    def release(self, holder: str) -> None:
        with self._lock:
            self._holders.pop(holder, None)

    # ── degrade order ──────────────────────────────────────────────────────
    def next_to_shed(self, active_work: "list[str]") -> str | None:
        """Under contention, the next piece of `active_work` to drop per the §9.1
        degrade order (pre_answer → screen → speculation_top1 → vlm). None when
        nothing sheddable is active."""
        try:
            present = set(active_work or [])
            for w in DEGRADE_ORDER:
                if w in present:
                    return w
            return None
        except Exception:  # noqa: BLE001
            return None

    # ── telemetry ──────────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            per_lane: dict[str, int] = {}
            for lane, _mb in self._holders.values():
                per_lane[lane] = per_lane.get(lane, 0) + 1
        return {"total_vram_mb": self.total_vram_mb, "reserved_mb": self.reserved_mb,
                "free_mb": self.free_mb, "live_sessions": self.live_sessions,
                "per_lane": per_lane}


# Process-wide scheduler, built from config on first use.
_scheduler: GpuScheduler | None = None
_build_lock = threading.RLock()


def scheduler() -> GpuScheduler:
    global _scheduler
    if _scheduler is None:
        with _build_lock:
            if _scheduler is None:
                total, cap, headroom = 24_000, 2, 1_000
                try:
                    from app.core.config_loader import cfg
                    total = int(getattr(cfg.gpu, "total_vram_mb", 24_000))
                    cap = int(getattr(cfg.gpu, "max_live_sessions", 2))
                    headroom = int(getattr(cfg.gpu, "bg_headroom_mb", 1_000))
                except Exception:  # noqa: BLE001
                    pass
                _scheduler = GpuScheduler(total_vram_mb=total,
                                          max_live_sessions=cap,
                                          bg_headroom_mb=headroom)
    return _scheduler


def reset_for_tests() -> None:
    global _scheduler
    with _build_lock:
        _scheduler = None


__all__ = ["REALTIME", "INTERACTIVE", "BACKGROUND", "RUN", "SHED_CLOUD",
           "DEFER", "REJECT", "DEGRADE_ORDER", "Admission", "GpuScheduler",
           "enabled", "scheduler", "reset_for_tests"]
