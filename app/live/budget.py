"""
Per-session economics (live-conversational-intelligence R22).

`SessionBudget` caps concurrent answers (and optionally total answers) per live
session so two rapid questions don't launch unbounded simultaneous generations
on a free tier. When provider rate-limit (429) pressure is detected it prefers a
faster/cheaper path, reusing the `intelligent-model-routing` penalties rather
than a new mechanism. Deterministic + fail-open: with the flag off (or a high
cap) behavior is today's unbounded concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionBudget:
    max_concurrent: int = 3          # simultaneous answers
    max_answers: int = 0             # total answers per session (0 = unlimited)
    active: int = 0
    started: int = 0

    @classmethod
    def from_config(cls) -> "SessionBudget":
        from app.core.config_loader import cfg
        return cls(
            max_concurrent=int(getattr(cfg.live, "max_concurrent_answers", 3) or 0),
            max_answers=int(getattr(cfg.live, "max_answers_per_session", 0) or 0),
        )

    def can_start(self) -> bool:
        if self.max_concurrent and self.active >= self.max_concurrent:
            return False
        if self.max_answers and self.started >= self.max_answers:
            return False
        return True

    def acquire(self) -> bool:
        """Reserve a concurrency slot. Returns False when at the cap (shed)."""
        if not self.can_start():
            return False
        self.active += 1
        self.started += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)

    def snapshot(self) -> dict:
        return {"active": self.active, "started": self.started,
                "max_concurrent": self.max_concurrent, "max_answers": self.max_answers}


def degrade_on_rate_limit() -> bool:
    """Best-effort: True when the router is currently under 429 pressure (so the
    live path should prefer the fast/cheaper answer). Reuses the
    intelligent-model-routing penalty state; never raises → False when unknown."""
    try:
        from app.llm import adaptive as _adaptive  # type: ignore
        checker = getattr(_adaptive, "is_rate_limited", None)
        if callable(checker):
            return bool(checker())
    except Exception:  # noqa: BLE001
        pass
    return False


# ---- per-session registry (in-process; no DB) -------------------------
_budgets: dict[str, SessionBudget] = {}


def get_budget(session_id: str) -> SessionBudget:
    b = _budgets.get(session_id)
    if b is None:
        b = SessionBudget.from_config()
        _budgets[session_id] = b
    return b


def forget_session(session_id: str) -> None:
    _budgets.pop(session_id, None)


# ── Per-stage Stage_Budget (R55) ───────────────────────────────────────
# A soft per-stage time budget (detection / retrieval / generation). When a
# stage overruns its budget, the live path DEGRADES that stage (skip optional
# enrichment / fall back to the fast answer) via adaptive-latency rather than
# blocking. Deterministic + fail-open.

# Default soft budgets in milliseconds per stage.
_STAGE_BUDGETS_MS = {
    "detection": 400.0,
    "retrieval": 800.0,
    "deliberation": 600.0,
    "generation": 6000.0,
}


def stage_budget_ms(stage: str) -> float:
    """Soft budget for a stage (ms). Config can override via
    `live.stage_budget_<stage>_ms`. Never raises → a sane default."""
    try:
        from app.core.config_loader import cfg
        key = f"stage_budget_{stage}_ms"
        override = getattr(cfg.live, key, None)
        if override:
            return float(override)
    except Exception:  # noqa: BLE001
        pass
    return _STAGE_BUDGETS_MS.get(stage, 1000.0)


def stage_over_budget(stage: str, elapsed_ms: float) -> bool:
    """True when a stage has overrun its soft budget (caller should degrade).
    Never raises → False."""
    try:
        return float(elapsed_ms) > stage_budget_ms(stage)
    except Exception:  # noqa: BLE001
        return False
