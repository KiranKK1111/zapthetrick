"""Live session model plan (vNext §2.7 F) — pin a primary + hot standby.

A Live interview must not re-evaluate the routing ladder mid-answer: that costs
latency and, worse, swaps the voice/style between turns (consistency is an
ACCURACY property for Live). So at session start we pin, per Live profile:

  * a **primary** and a **hot standby** — both gauntlet-probed-healthy (§2.5),
    the standby preferring a DIFFERENT provider so one provider outage can't take
    both down (same canonical model when possible, for voice consistency);
  * both **pre-connected** (§3.4) — the caller warms them from the pinned list;
  * the session's **expected spend reserved** against their per-key ledgers
    (§2.7 D) for the session's duration, so routine traffic can't eat the budget
    mid-interview.

`next_model(...)` is the failover: the primary while it's healthy, the standby on
the primary's FIRST failure (zero ladder re-evaluation), and `None` — fall to the
ordinary never-empty ladder — only when both are down. Otherwise sticky.

Self-contained + fail-open. Full Live pre-flight wiring lands with Stage 6; this
module owns the plan, the reservation, and the failover decision. The caller
supplies the ranked candidate list (from the router), so this module imports no
DB and stays inside `llm`.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Default requests to reserve for a session when the caller doesn't estimate.
_DEFAULT_RESERVE = 60


@dataclass(frozen=True)
class Candidate:
    """A routable option the caller hands the planner (from the router's ranked
    pool). `cid_key` groups same-model-different-provider; `model_db_id`/`key_id`
    identify the concrete row + key for pinning and reservation."""
    cid_key: str
    provider: str
    model_db_id: int | None = None
    key_id: int | None = None


@dataclass(frozen=True)
class PinnedModel:
    cid_key: str
    provider: str
    model_db_id: int | None
    key_id: int | None

    @classmethod
    def of(cls, c: Candidate) -> "PinnedModel":
        return cls(c.cid_key, c.provider, c.model_db_id, c.key_id)


@dataclass
class LiveSessionPlan:
    session_id: str
    profile: str                       # live_answer | live_code
    primary: PinnedModel
    standby: PinnedModel | None
    reserved: int                      # requests reserved per pinned model
    created_at: float
    primary_failed: bool = False
    standby_failed: bool = False
    pinned: list[PinnedModel] = field(default_factory=list)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, "live_plan", False))
    except Exception:  # noqa: BLE001
        return False


def _healthy(c: Candidate) -> bool:
    """A candidate is eligible for pinning only if the gauntlet hasn't
    quarantined it (unproven models never anchor a Live session). Gauntlet off
    → everything is eligible."""
    try:
        from app.llm import gauntlet as _g
        return not _g.is_quarantined(c.cid_key, c.provider)
    except Exception:  # noqa: BLE001
        return True


class LivePlanner:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._plans: dict[str, LiveSessionPlan] = {}
        self._now = now or time.time

    def get(self, session_id: str) -> LiveSessionPlan | None:
        return self._plans.get(session_id)

    def plan(self, session_id: str, profile: str,
             candidates: list[Candidate], *,
             expected_requests: int = _DEFAULT_RESERVE,
             reserve: bool = True) -> LiveSessionPlan | None:
        """Pin primary + standby from the ranked `candidates` (best-first). The
        standby prefers a DIFFERENT provider (failover diversity), same canonical
        model when possible (voice consistency). Reserves `expected_requests`
        against each pinned model's ledger (§2.7 D). Returns the plan, or None if
        no healthy candidate exists (caller uses the ordinary ladder). Re-planning
        the same session releases the old reservation first."""
        self.release(session_id)
        healthy = [c for c in candidates if _healthy(c)]
        if not healthy:
            return None
        primary = healthy[0]
        # Standby: the best remaining candidate on a DIFFERENT provider (so a
        # provider outage doesn't fell both); among those, prefer the SAME model.
        others = [c for c in healthy[1:] if c.provider != primary.provider]
        same_model = [c for c in others if c.cid_key == primary.cid_key]
        standby_c = (same_model[0] if same_model
                     else (others[0] if others else None))

        pins = [PinnedModel.of(primary)]
        if standby_c is not None:
            pins.append(PinnedModel.of(standby_c))
        plan = LiveSessionPlan(
            session_id=session_id, profile=profile,
            primary=PinnedModel.of(primary),
            standby=PinnedModel.of(standby_c) if standby_c else None,
            reserved=max(0, int(expected_requests)),
            created_at=self._now(), pinned=pins)
        if reserve and plan.reserved > 0:
            self._reserve(plan)
        self._plans[session_id] = plan
        return plan

    def next_model(self, session_id: str, *,
                   primary_failed: bool = False) -> PinnedModel | None:
        """The model to use for the next Live turn. Sticky to the primary; on the
        primary's FIRST failure hand off to the standby with ZERO re-evaluation;
        when both are down, return None so the caller drops to the ordinary
        ladder. Records the failure so the handoff persists across turns."""
        plan = self._plans.get(session_id)
        if plan is None:
            return None
        if primary_failed:
            plan.primary_failed = True
        if not plan.primary_failed:
            return plan.primary
        if plan.standby is not None and not plan.standby_failed:
            return plan.standby
        return None                     # deeper failure → ordinary ladder

    def note_standby_failed(self, session_id: str) -> None:
        plan = self._plans.get(session_id)
        if plan is not None:
            plan.standby_failed = True

    def release(self, session_id: str) -> None:
        """End a session: return its reservations to the ledger."""
        plan = self._plans.pop(session_id, None)
        if plan is None or plan.reserved <= 0:
            return
        try:
            from app.llm.quota_plan import quota_planner
            qp = quota_planner()
            for pin in plan.pinned:
                qp.release(pin.provider, pin.key_id, plan.reserved)
        except Exception:  # noqa: BLE001
            pass

    def _reserve(self, plan: LiveSessionPlan) -> None:
        try:
            from app.llm.quota_plan import quota_planner
            qp = quota_planner()
            for pin in plan.pinned:
                qp.reserve(pin.provider, pin.key_id, plan.reserved)
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        for sid in list(self._plans):
            self.release(sid)


_planner = LivePlanner()


def live_planner() -> LivePlanner:
    return _planner


def reset_for_tests() -> None:
    _planner.clear()
    _planner._plans.clear()


__all__ = ["Candidate", "PinnedModel", "LiveSessionPlan", "LivePlanner",
           "live_planner", "enabled", "reset_for_tests"]
