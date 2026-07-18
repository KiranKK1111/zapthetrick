"""Architecture Health Dashboard (roadmap Phase 1 #6).

One live snapshot of the system's operational health — cache hit rate, provider
quota headroom, background jobs, and the failure-recovery ledger — composed from
the in-process metric sources that already exist. Each source is pulled
fail-open, so a missing/erroring source degrades to an empty section instead of
breaking the dashboard.
"""
from __future__ import annotations


def _cache() -> dict:
    try:
        from app.llm.cache import stats
        return stats()
    except Exception:  # noqa: BLE001
        return {}


def _providers() -> list:
    try:
        from app.llm.quota_manager import quota_manager
        return quota_manager().snapshot()
    except Exception:  # noqa: BLE001
        return []


def _jobs() -> dict:
    try:
        from app.obs.jobs import jobs
        js = jobs().snapshot()
        return {
            "running": sum(1 for j in js if j.get("status") == "running"),
            "total": len(js),
            "recent": js[:8],
        }
    except Exception:  # noqa: BLE001
        return {}


def _failures() -> dict:
    try:
        from app.obs.failure_kb import stats
        return stats()
    except Exception:  # noqa: BLE001
        return {}


def _router_cost() -> dict:
    """Router cost proxy: estimated tokens/$ per routed call + provider request
    usage. (Cost is an estimate — tokens, not billed $ — as elsewhere.)"""
    try:
        from app.obs.metrics import router_cost_snapshot
        return router_cost_snapshot()
    except Exception:  # noqa: BLE001
        return {}


def _retrieval_relevance() -> dict:
    """Mean top-k retrieval relevance across recent retrievals."""
    try:
        from app.obs.metrics import retrieval_relevance_snapshot
        return retrieval_relevance_snapshot()
    except Exception:  # noqa: BLE001
        return {}


def _verifier() -> dict:
    """Verifier failure rate (answer + artifact validation outcomes)."""
    try:
        from app.obs.metrics import verifier_snapshot
        return verifier_snapshot()
    except Exception:  # noqa: BLE001
        return {}


def snapshot() -> dict:
    """The whole dashboard in one call."""
    return {
        "cache": _cache(),
        "providers": _providers(),
        "jobs": _jobs(),
        "failures": _failures(),
        # Phase 1 #6 — the three fields the roadmap flagged missing:
        "router_cost": _router_cost(),
        "retrieval_relevance": _retrieval_relevance(),
        "verifier": _verifier(),
    }


__all__ = ["snapshot"]
