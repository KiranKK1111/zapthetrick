"""Live pre-flight session-plan wiring (vNext §4.6 / §2.7 F, Stage 6 Component B).

At Live session start this pins the Stage-5 model plan and warms it, so the
pre-flight board's `model_plan` check (Component A) goes green in a real session
and the first turn fires on an already-open connection:

  1. **rank candidates** for the Live profile (the router's real pick as primary,
     plus the SAME canonical model on other providers as standby options);
  2. **`live_plan.plan(...)`** — pin primary + hot standby (gauntlet-healthy) and
     RESERVE the session's expected spend against D's per-key ledgers (§2.7);
  3. **warm** the pinned providers (§3.4 pre-connect), fire-and-forget so session
     creation stays fast.

The candidate source is INJECTABLE (`candidate_fn`) so the orchestration is
unit-tested with no router/DB; the default reuses `select_route` for a
quality-correct primary and `identity.build_provider_index` for same-model
standbys. Flag-gated (`routing.live_plan`, default OFF → no plan, ordinary
ladder every turn) and fully fail-open — any error yields `None` (the session
just runs unplanned). Discharges the Stage-5 Live-pre-flight follow-on.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from app.llm import live_plan as _lp
from app.llm.live_plan import Candidate, LiveSessionPlan

log = logging.getLogger(__name__)

# Requests to reserve for a session against each pinned model's ledger.
_DEFAULT_RESERVE = 60

CandidateFn = Callable[[str], "list[Candidate] | Awaitable[list[Candidate]]"]


def enabled() -> bool:
    """Component B rides the Stage-5 F flag — no plan unless `routing.live_plan`."""
    return _lp.enabled()


async def plan_live_session(
    session_id: str, profile: str = "live_answer", *,
    candidate_fn: CandidateFn | None = None, warm: bool = True,
    expected_requests: int = _DEFAULT_RESERVE,
) -> LiveSessionPlan | None:
    """Pin + reserve + warm a Live session's model plan. Returns the plan, or
    None when disabled / no healthy candidate / any error (fail-open)."""
    if not enabled():
        return None
    try:
        candidates = await _resolve_candidates(candidate_fn, profile)
        if not candidates:
            return None
        plan = _lp.live_planner().plan(
            session_id, profile, candidates,
            expected_requests=expected_requests)
        if plan is not None and warm:
            _warm_pins(plan)
        return plan
    except Exception as exc:  # noqa: BLE001 — planning never blocks a session
        log.info("live session plan failed (%s): %s", session_id, exc)
        return None


async def _resolve_candidates(candidate_fn: CandidateFn | None,
                              profile: str) -> list[Candidate]:
    if candidate_fn is None:
        return await _router_candidates(profile)
    res = candidate_fn(profile)
    if inspect.isawaitable(res):
        res = await res
    return list(res or [])


async def _router_candidates(profile: str, *, limit: int = 4) -> list[Candidate]:
    """Default candidate source: the router's real primary pick, plus the SAME
    canonical model on OTHER providers (from the seed catalog) as standby
    options — exactly the diversity `live_plan` wants (voice consistency +
    provider-failover). Fail-open → [] so a router hiccup just skips planning."""
    try:
        from app.llm import catalog
        from app.llm.identity import (
            build_provider_index, canonicalize, providers_for)
        from app.llm.router import select_route

        # Live profiles want quality → route at the strong tier.
        decision = await select_route(task_profile=profile, difficulty="hard")
        route = getattr(decision, "route", None)
        if route is None:
            return []
        prim_cid = canonicalize(route.platform, route.model_id)
        out = [Candidate(prim_cid.key(), route.platform,
                         route.model_db_id, route.key_id)]
        # Same-identity providers from the curated seed list (a stable, DB-free
        # source of "who else serves this model").
        rows = [(r[0], r[1], None) for r in catalog.MODEL_SEED]
        seen = {route.platform}
        for platform, _model_id, _ref in providers_for(
                prim_cid, build_provider_index(rows)):
            if platform in seen:
                continue
            seen.add(platform)
            out.append(Candidate(prim_cid.key(), platform, None, None))
            if len(out) >= limit:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        log.info("router live-candidates failed: %s", exc)
        return []


def _warm_pins(plan: LiveSessionPlan) -> None:
    """Fire-and-forget pre-connect for each pinned provider (§3.4) so session
    creation isn't blocked on network warmth."""
    for pin in plan.pinned:
        try:
            asyncio.ensure_future(_warm_provider(pin.provider))
        except Exception:  # noqa: BLE001 — no loop / scheduling error → skip
            pass


async def _warm_provider(platform: str) -> None:
    """Open a pooled connection to `platform`'s base host so the first Live turn
    skips DNS+TLS. A 4xx/401 still warms the socket. Never raises."""
    try:
        from app.core.http_pool import get_http_client
        from app.llm.catalog import get_provider_spec
        spec = get_provider_spec(platform)
        base = (getattr(spec, "base_url", "") or "") if spec else ""
        if not base or "{" in base:
            return
        client = get_http_client()
        try:
            await client.get(base.rstrip("/"), timeout=2.0)
        except Exception:  # noqa: BLE001 — a non-200 still warms the socket
            pass
    except Exception:  # noqa: BLE001
        pass


def release_live_session(session_id: str) -> None:
    """End-of-session hook: release the plan's ledger reservation."""
    try:
        _lp.live_planner().release(session_id)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["enabled", "plan_live_session", "release_live_session"]
