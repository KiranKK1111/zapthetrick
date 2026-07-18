"""General goal negotiation (roadmap Phase 5 #22).

`capabilities/registry.negotiate_format` already downgrades an impossible
DOCUMENT FORMAT to the closest one we can render. This generalises that idea to
whole GOALS: when a request asks for something this deployment cannot do (deploy
to prod, run a live web search with search disabled, produce a video), pick the
closest ACHIEVABLE goal and say why — instead of promising a deliverable we
can't produce or failing opaquely.

A goal is achievable when its required capability is present. Each goal has a
fallback ladder toward progressively more modest, always-achievable goals
(ultimately "explain / describe"). Capability presence is injected (a callable)
or read from the runtime capability registry; deterministic + fail-open.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Goal ids.
DEPLOY = "deploy"
WEB_RESEARCH = "web_research"
RUN_CODE = "run_code"
BUILD_DOC = "build_document"
GENERATE_MEDIA = "generate_media"
WRITE_CODE = "write_code"
DRAFT_PLAN = "draft_plan"
EXPLAIN = "explain"          # the always-achievable floor

# goal → (required capability key, fallback goal, why the fallback still helps)
_LADDER: dict[str, tuple[str | None, str, str]] = {
    DEPLOY:        ("deploy",       RUN_CODE,       "can't deploy here; I can build and run it instead"),
    WEB_RESEARCH:  ("web_search",   EXPLAIN,        "web search is unavailable; answering from knowledge"),
    RUN_CODE:      ("sandbox",      WRITE_CODE,     "no execution sandbox; I'll write the code instead"),
    GENERATE_MEDIA:("media",        DRAFT_PLAN,     "can't render media; I'll draft a script/storyboard"),
    BUILD_DOC:     ("document",     DRAFT_PLAN,     "can't render that document; I'll outline it"),
    WRITE_CODE:    (None,           WRITE_CODE,     "writing code (always available)"),
    DRAFT_PLAN:    (None,           DRAFT_PLAN,     "drafting a plan (always available)"),
    EXPLAIN:       (None,           EXPLAIN,        "explaining (always available)"),
}


@dataclass
class NegotiatedGoal:
    requested: str
    achievable: str
    downgraded: bool
    reason: str

    def as_dict(self) -> dict:
        return {"requested": self.requested, "achievable": self.achievable,
                "downgraded": self.downgraded, "reason": self.reason}


def _default_available(capability: str) -> bool:
    """Default capability check when the caller injects none.

    Capabilities this deployment structurally lacks ('deploy', 'media') are
    unavailable; everything else is assumed available (unknown → available, so we
    never over-restrict). Callers that HAVE a live capability snapshot (the
    clarify/policy layer, which owns the capability registry) pass a concrete
    `available` callable — keeping this module free of a cross-package import to
    `capabilities` (import-boundary guardrail)."""
    return capability not in ("deploy", "media")


def negotiate_goal(
    goal: str,
    *,
    available: Callable[[str], bool] | None = None,
    _depth: int = 0,
) -> NegotiatedGoal:
    """Return the closest achievable goal to `goal`.

    Walks the fallback ladder until a goal whose required capability is present
    (or a floor goal with no requirement). Fail-open: unknown goal → itself,
    not downgraded.
    """
    try:
        avail = available or _default_available
        g = (goal or "").strip().lower()
        if g not in _LADDER:
            return NegotiatedGoal(goal, goal, False, "unknown goal — passed through")
        req_cap, fallback, why = _LADDER[g]
        if req_cap is None or avail(req_cap):
            return NegotiatedGoal(goal, g, False, "achievable")
        if _depth >= len(_LADDER):        # cycle guard
            return NegotiatedGoal(goal, EXPLAIN, True, why)
        deeper = negotiate_goal(fallback, available=avail, _depth=_depth + 1)
        return NegotiatedGoal(goal, deeper.achievable, True, why)
    except Exception:  # noqa: BLE001
        return NegotiatedGoal(goal, goal, False, "goal negotiation error — passed through")


__all__ = [
    "NegotiatedGoal", "negotiate_goal",
    "DEPLOY", "WEB_RESEARCH", "RUN_CODE", "BUILD_DOC", "GENERATE_MEDIA",
    "WRITE_CODE", "DRAFT_PLAN", "EXPLAIN",
]
