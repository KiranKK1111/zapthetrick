"""Recovery Planner (roadmap Phase 4 #8).

Turns a failure into a concrete, *bounded* recovery decision — consuming the
Phase 1 failure taxonomy's per-class recovery strategy. This is the anti-pattern
guard against blind retries (rule #10): every retry is classified, budgeted, and
backed off, and non-retryable classes escalate/abort instead of looping.

Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.obs import failure_taxonomy as ft
from app.obs.failure_taxonomy import FailureClass, Recovery

# Recovery actions that mean "try again" (in some form) vs terminal ones.
_RETRY_FAMILY = {
    Recovery.RETRY, Recovery.RETRY_DIFFERENT, Recovery.COOLDOWN_WAIT,
    Recovery.REPAIR, Recovery.REPLAN,
}

# String form of the retry family — the vocabulary callers (the LLM engine) act
# on, since a plan carries `action` as a Recovery *value*, not the enum.
RETRY_ACTIONS: frozenset[str] = frozenset(r.value for r in _RETRY_FAMILY)

# Actions that mean "don't come back to the same model/key". REPAIR/REPLAN have
# no in-engine meaning (the engine can't repair an artifact), so at the routing
# layer they degrade to "try somewhere else" — the closest executable recovery.
DIFFERENT_ROUTE_ACTIONS: frozenset[str] = frozenset({
    Recovery.RETRY_DIFFERENT.value, Recovery.REPAIR.value, Recovery.REPLAN.value,
})


def is_retry_action(action: str | None) -> bool:
    """True when `action` is a 'try again' strategy (vs a terminal one)."""
    return bool(action) and action in RETRY_ACTIONS


def wants_different_route(action: str | None) -> bool:
    """True when the strategy says the retry must go to a DIFFERENT model/key."""
    return bool(action) and action in DIFFERENT_ROUTE_ACTIONS


@dataclass(frozen=True)
class RecoveryPlan:
    failure_id: str
    action: str            # the Recovery strategy value
    should_retry: bool     # honor the attempt budget
    attempt: int
    max_attempts: int
    backoff_ms: int
    rationale: str
    learned_action: str | None = None  # history-backed recovery, if the KB knows one

    @property
    def terminal(self) -> bool:
        """The failure class itself can't be retried away (escalate / abort /
        degrade / clarify / skip) — distinct from having merely spent its budget."""
        return not is_retry_action(self.action)

    @property
    def budget_exhausted(self) -> bool:
        """Retryable in principle, but this attempt is the last one allowed."""
        return not self.should_retry and not self.terminal

    @property
    def effective_action(self) -> str:
        """The action to actually TAKE: the KB's history-backed recovery when it
        knows one that's executable, else the taxonomy default. This is what the
        outcome must be recorded against (`failure_kb.record_outcome`) — recording
        the default while executing the learned one would poison the KB."""
        if self.learned_action and is_retry_action(self.learned_action):
            return self.learned_action
        return self.action


def _resolve(failure) -> FailureClass:
    if isinstance(failure, FailureClass):
        return failure
    if isinstance(failure, str):
        return ft.get(failure) or ft.TAXONOMY["internal_error"]
    if isinstance(failure, BaseException):
        return ft.classify_exception(failure)
    return ft.TAXONOMY["internal_error"]


def _backoff_ms(recovery: Recovery, attempt: int) -> int:
    """Escalating backoff for rate-limit waits; a small constant elsewhere;
    zero for terminal actions."""
    if recovery is Recovery.COOLDOWN_WAIT:
        return min(60_000, 1_000 * (2 ** max(0, attempt - 1)))  # 1s,2s,4s… capped 60s
    if recovery in _RETRY_FAMILY:
        return 250
    return 0


def backoff_for(action: str | None, attempt: int) -> int:
    """Backoff (ms) for a strategy *value* on `attempt`. Same curve as a plan's
    own `backoff_ms`, but recomputable when the action actually executed differs
    from the taxonomy default (see `RecoveryPlan.effective_action`). Never raises."""
    try:
        return _backoff_ms(Recovery(action), max(1, int(attempt)))
    except Exception:  # noqa: BLE001 — unknown action → no backoff
        return 0


def plan_recovery(failure, *, attempt: int = 1, max_attempts: int = 3) -> RecoveryPlan:
    """Decide how to recover from `failure` on this `attempt`. `failure` may be a
    FailureClass, a taxonomy id string, or a raw exception."""
    try:
        fc = _resolve(failure)
        in_family = fc.recovery in _RETRY_FAMILY
        should_retry = in_family and attempt < max_attempts
        backoff = _backoff_ms(fc.recovery, attempt) if should_retry else 0
        # Consult the Failure KB (Phase 7 #3): if history shows a recovery that
        # actually works for this failure, surface it (non-destructive hint).
        learned = None
        try:
            from app.obs import failure_kb
            learned = failure_kb.best_recovery(fc.id)
        except Exception:  # noqa: BLE001
            learned = None
        if should_retry:
            why = f"{fc.id}: {fc.recovery.value} (attempt {attempt}/{max_attempts})"
        elif in_family:
            why = f"{fc.id}: retry budget exhausted ({attempt}/{max_attempts}) → give up"
        else:
            why = f"{fc.id}: {fc.recovery.value} (terminal — no retry)"
        if learned and learned != fc.recovery.value:
            why += f"; KB suggests '{learned}' from history"
        return RecoveryPlan(
            failure_id=fc.id, action=fc.recovery.value, should_retry=should_retry,
            attempt=attempt, max_attempts=max_attempts, backoff_ms=backoff,
            rationale=why, learned_action=learned,
        )
    except Exception:  # noqa: BLE001 — recovery planning must never itself fail
        return RecoveryPlan("internal_error", Recovery.ESCALATE.value, False,
                            attempt, max_attempts, 0, "recovery planner error → escalate")


__all__ = [
    "RecoveryPlan", "plan_recovery", "backoff_for",
    "is_retry_action", "wants_different_route",
    "RETRY_ACTIONS", "DIFFERENT_ROUTE_ACTIONS",
]
