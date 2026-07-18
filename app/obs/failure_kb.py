"""Failure Knowledge Base (roadmap Phase 7 #3).

Closes the reliability loop across phases: the taxonomy (Phase 1) *names* a
failure, the recovery planner (Phase 4) proposes a strategy, and this KB *learns
from outcomes* which recovery actually works for each failure class — so
recurring failures get a proven, history-backed recovery instead of only the
static default.

Records outcomes per (failure_id, recovery_action); recommends the action with
the best observed success rate (Laplace-smoothed). Deterministic + fail-open;
process-global (single-user).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _ActionStat:
    successes: int = 0
    failures: int = 0

    @property
    def attempts(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return (self.successes + 1) / (self.attempts + 2)  # Laplace


@dataclass
class FailureRecord:
    failure_id: str
    occurrences: int = 0
    actions: dict[str, _ActionStat] = field(default_factory=dict)


_kb: dict[str, FailureRecord] = {}


def _rec(failure_id: str) -> FailureRecord:
    r = _kb.get(failure_id)
    if r is None:
        r = FailureRecord(failure_id)
        _kb[failure_id] = r
    return r


def record_occurrence(failure_id: str) -> None:
    """Note that a failure class happened (independent of any recovery)."""
    try:
        _rec(failure_id).occurrences += 1
    except Exception:  # noqa: BLE001
        pass


def record_outcome(failure_id: str, action: str, success: bool) -> None:
    """Note that `action` did/didn't recover `failure_id`."""
    try:
        r = _rec(failure_id)
        st = r.actions.get(action)
        if st is None:
            st = _ActionStat()
            r.actions[action] = st
        if success:
            st.successes += 1
        else:
            st.failures += 1
    except Exception:  # noqa: BLE001
        pass


def best_recovery(failure_id: str, *, min_attempts: int = 2) -> str | None:
    """The recovery action with the best observed success rate for this failure,
    or None if there isn't enough history to recommend one."""
    try:
        r = _kb.get(failure_id)
        if not r or not r.actions:
            return None
        eligible = [(a, s) for a, s in r.actions.items() if s.attempts >= min_attempts]
        if not eligible:
            return None
        return max(eligible, key=lambda kv: kv[1].success_rate)[0]
    except Exception:  # noqa: BLE001
        return None


def known(failure_id: str) -> bool:
    return failure_id in _kb


def stats() -> dict:
    return {
        fid: {
            "occurrences": r.occurrences,
            "actions": {a: {"success_rate": round(s.success_rate, 3),
                            "attempts": s.attempts}
                        for a, s in r.actions.items()},
        }
        for fid, r in _kb.items()
    }


def reset() -> None:
    _kb.clear()


__all__ = [
    "FailureRecord", "record_occurrence", "record_outcome",
    "best_recovery", "known", "stats", "reset",
]
