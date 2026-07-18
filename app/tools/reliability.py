"""Tool / Capability Reliability Scores (roadmap Phase 5 #11).

Tracks each tool's real success/failure history and exposes a smoothed
reliability score, so the planner can prefer reliable tools and avoid ones that
are currently degraded — instead of treating every registered tool as equally
trustworthy. Complements the router's per-model `learning.learned_success`
(the model-behavior half of #11); this is the tool half.

Laplace-smoothed so an unknown tool starts at 0.5 and converges to its true
rate. Deterministic + fail-open; process-global (single-user, single process).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolStat:
    name: str
    successes: int = 0
    failures: int = 0

    @property
    def attempts(self) -> int:
        return self.successes + self.failures

    @property
    def reliability(self) -> float:
        # (s+1)/(s+f+2): unknown → 0.5, converges to the true success rate.
        return round((self.successes + 1) / (self.attempts + 2), 4)


_stats: dict[str, ToolStat] = {}


def record(name: str, success: bool) -> None:
    try:
        st = _stats.get(name)
        if st is None:
            st = ToolStat(name)
            _stats[name] = st
        if success:
            st.successes += 1
        else:
            st.failures += 1
    except Exception:  # noqa: BLE001 — telemetry must never break a tool call
        pass


def reliability(name: str) -> float:
    """Smoothed success rate in [0,1]; 0.5 for a never-seen tool."""
    st = _stats.get(name)
    return st.reliability if st else 0.5


def is_degraded(name: str, *, threshold: float = 0.4, min_attempts: int = 3) -> bool:
    """True when a tool has enough history AND its reliability is below the
    threshold — so a single early failure doesn't condemn a tool."""
    st = _stats.get(name)
    return bool(st and st.attempts >= min_attempts and st.reliability < threshold)


def rank(names: list[str]) -> list[str]:
    """Tool names sorted by reliability, most reliable first (stable)."""
    return sorted(names, key=lambda n: (-reliability(n), names.index(n)))


def snapshot() -> dict:
    return {n: {"reliability": s.reliability, "attempts": s.attempts}
            for n, s in _stats.items()}


def reset() -> None:
    _stats.clear()


__all__ = ["ToolStat", "record", "reliability", "is_degraded", "rank",
           "snapshot", "reset"]
