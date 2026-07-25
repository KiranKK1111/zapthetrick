"""Ops resilience (vNext §6.4 / §11.3 / §11.4, Stage 9 Component E).

The deterministic core of three ops capabilities whose EXECUTION is deploy/infra
but whose DECISION logic is pure and testable:

  * **Zero-downtime drain (§6.4)** — `DrainController`: stop accepting new turns,
    let in-flight ones finish (≤30 s), report drained. The `/admin/drain` handler
    + entrypoint drive it; the state machine + deadline maths live here.
  * **Data lifecycle (§11.3)** — retention-as-DATA (`RETENTION`) + a ref-counted
    blob **GC planner** (`plan_gc`): a referenced blob is kept; an unreferenced
    one is purged per its kind's policy (eval 90 d, screen-state 24 h, pre-answers
    session-scoped, voice never).
  * **Pod resilience (§11.4)** — `ResurrectionMonitor`: N consecutive failed
    health probes → recreate-from-template. The RunPod API call is the seam; the
    "when to trigger" decision is here.

All times are INJECTED (`now`) so the logic is deterministic + resume-safe. Pure +
fail-open. Flag-gated (`deploy.drain`, `ops.gc`, `ops.redirector`, default OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Drain states.
ACCEPTING = "accepting"
DRAINING = "draining"
DRAINED = "drained"

_DAY = 86400.0

# Retention-as-DATA (§11.3): kind → retain seconds.
#   > 0  age-based TTL      (purge when older, if unreferenced)
#   -1   never purge        (keep forever — e.g. voice consent artifacts)
#    0   session-scoped     (purge when the owning session ends)
RETENTION: dict[str, float] = {
    "eval": 90 * _DAY,
    "screen_state": _DAY,          # 24 h
    "pre_answer": 0.0,             # session-scoped
    "voice": -1.0,                 # never
    "artifact": 30 * _DAY,         # generated documents
    "upload": 30 * _DAY,
}
_DEFAULT_TTL = 30 * _DAY


def _deploy_drain() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.deploy, "drain", False))
    except Exception:  # noqa: BLE001
        return False


def _ops_gc() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.ops, "gc", False))
    except Exception:  # noqa: BLE001
        return False


def _drain_deadline() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.deploy, "drain_deadline_s", 30.0) or 30.0)
    except Exception:  # noqa: BLE001
        return 30.0


# --------------------------------------------------------------------------- #
# Zero-downtime drain (§6.4)
# --------------------------------------------------------------------------- #
@dataclass
class DrainController:
    state: str = ACCEPTING
    started_at: float | None = None
    inflight: set = field(default_factory=set)

    def begin(self, *, now: float) -> None:
        """Enter DRAINING: stop accepting new turns, keep the in-flight ones."""
        if self.state == ACCEPTING:
            self.state = DRAINING
            self.started_at = now

    def should_accept(self) -> bool:
        """New turns are accepted ONLY while ACCEPTING. When the flag is off,
        always accept (byte-identical)."""
        if not _deploy_drain():
            return True
        return self.state == ACCEPTING

    def add_inflight(self, turn_id) -> None:
        self.inflight.add(turn_id)

    def finish_inflight(self, turn_id) -> None:
        self.inflight.discard(turn_id)
        if self.state == DRAINING and not self.inflight:
            self.state = DRAINED

    def inflight_count(self) -> int:
        return len(self.inflight)

    def is_drained(self) -> bool:
        return self.state == DRAINED or (
            self.state == DRAINING and not self.inflight)

    def deadline_exceeded(self, *, now: float, deadline_s: float | None = None) -> bool:
        """Whether draining has run past its deadline (force restart even with
        stragglers). False until draining starts. Never raises."""
        try:
            if self.state != DRAINING or self.started_at is None:
                return False
            dl = _drain_deadline() if deadline_s is None else deadline_s
            return (now - self.started_at) >= dl
        except Exception:  # noqa: BLE001
            return False

    def can_restart(self, *, now: float) -> bool:
        """Safe to restart when drained OR the deadline passed."""
        return self.is_drained() or self.deadline_exceeded(now=now)

    def to_dict(self) -> dict:
        return {"state": self.state, "inflight": self.inflight_count(),
                "drained": self.is_drained()}


# --------------------------------------------------------------------------- #
# Data lifecycle (§11.3) — retention + ref-counted blob GC
# --------------------------------------------------------------------------- #
def retention_for(kind: str) -> float:
    return RETENTION.get((kind or "").strip().lower(), _DEFAULT_TTL)


def should_purge(kind: str, *, age_s: float, referenced: bool,
                 session_active: bool = False) -> bool:
    """Whether a blob should be purged. Ref-counted: a REFERENCED blob is always
    kept. Else its kind's retention policy decides — never(-1) keep, session(0)
    purge once the session ends, TTL(>0) purge when older. Never raises → keep
    (the safe default: never delete on uncertainty)."""
    try:
        if referenced:
            return False
        ttl = retention_for(kind)
        if ttl < 0:
            return False                       # never purge
        if ttl == 0:
            return not session_active          # session-scoped
        return age_s > ttl
    except Exception:  # noqa: BLE001
        return False


@dataclass
class GCPlan:
    purge: list = field(default_factory=list)   # blob ids to delete
    keep: list = field(default_factory=list)
    freed_bytes: int = 0

    def to_dict(self) -> dict:
        return {"purge": list(self.purge), "keep": list(self.keep),
                "freed_bytes": self.freed_bytes, "count": len(self.purge)}


def plan_gc(blobs, *, now: float) -> GCPlan:
    """Plan a ref-counted blob GC pass. Each blob is duck-typed
    {id, kind, created_at, refs, size, session_active}. Returns which to purge vs
    keep + the bytes freed. NEVER purges when GC is disabled (empty plan). Never
    raises."""
    plan = GCPlan()
    if not _ops_gc():
        return plan
    try:
        for b in blobs or ():
            bid = _get(b, "id")
            kind = _get(b, "kind", "artifact")
            created_raw = _get(b, "created_at", now)
            created = float(now if created_raw is None else created_raw)
            refs = int(_get(b, "refs", 0) or 0)
            size = int(_get(b, "size", 0) or 0)
            session_active = bool(_get(b, "session_active", False))
            age = max(0.0, now - created)
            if should_purge(kind, age_s=age, referenced=refs > 0,
                            session_active=session_active):
                plan.purge.append(bid)
                plan.freed_bytes += size
            else:
                plan.keep.append(bid)
        return plan
    except Exception:  # noqa: BLE001
        return GCPlan()


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
# Pod resilience (§11.4) — auto-resurrection
# --------------------------------------------------------------------------- #
@dataclass
class ResurrectionMonitor:
    """Tracks consecutive health-probe failures; triggers recreate-from-template
    after `threshold` in a row (reset on any success)."""
    consecutive_failures: int = 0
    threshold: int = 0                 # 0 → resolved from config lazily
    triggered: bool = False

    def _threshold(self) -> int:
        if self.threshold:
            return self.threshold
        try:
            from app.core.config_loader import cfg
            return max(1, int(getattr(cfg.ops, "resurrect_after_failures", 3) or 3))
        except Exception:  # noqa: BLE001
            return 3

    def record_probe(self, ok: bool) -> None:
        """Record a health-probe result. A success resets the counter."""
        if ok:
            self.consecutive_failures = 0
            self.triggered = False
        else:
            self.consecutive_failures += 1

    def should_resurrect(self) -> bool:
        """True once failures reach the threshold AND we haven't already fired
        (so resurrection triggers once per outage). Requires the redirector flag."""
        try:
            from app.core.config_loader import cfg
            if not bool(getattr(cfg.ops, "redirector", False)):
                return False
        except Exception:  # noqa: BLE001
            return False
        if self.triggered:
            return False
        return self.consecutive_failures >= self._threshold()

    def mark_triggered(self) -> None:
        self.triggered = True

    def to_dict(self) -> dict:
        return {"consecutive_failures": self.consecutive_failures,
                "threshold": self._threshold(), "triggered": self.triggered}


__all__ = ["ACCEPTING", "DRAINING", "DRAINED", "DrainController", "RETENTION",
           "retention_for", "should_purge", "GCPlan", "plan_gc",
           "ResurrectionMonitor"]
