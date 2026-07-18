"""Unified stream budget controller (roadmap Phase 6 #17).

The deadlines that govern a streaming turn were scattered across three modules
(`perceived/budget.py`, `context_budget.py`, `live/budget.py`) with no single
owner of the one question a streaming loop actually asks: *"by when must the user
see the first visible update, and when have I run out of time?"*

:class:`StreamBudget` unifies them into ONE per-task controller with three
milestones:

* **ack_threshold_s**   — if nothing is on screen by here, emit an ``ack`` frame;
* **first_visible_s**   — the hard first-meaningful-update deadline (TTFMU);
* **total_s**           — the whole-turn ceiling.

It reads the existing config keys (fail-open defaults matching config.yaml) so it
does not introduce new tunables, and every method is a pure comparison against an
elapsed-seconds value the caller measures with a monotonic clock.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamBudget:
    ack_threshold_s: float
    first_visible_s: float
    total_s: float

    # -- milestone checks (caller passes monotonic elapsed seconds) ----------
    def should_ack(self, elapsed_s: float, *, first_seen: bool) -> bool:
        """True when we should send an ``ack`` (nothing visible past the ack
        threshold). No-op once the first token/block has been seen."""
        return (not first_seen and self.ack_threshold_s > 0
                and elapsed_s >= self.ack_threshold_s)

    def first_visible_overdue(self, elapsed_s: float, *,
                              first_seen: bool) -> bool:
        """True when the first-meaningful-update deadline has passed unmet."""
        return (not first_seen and self.first_visible_s > 0
                and elapsed_s >= self.first_visible_s)

    def exhausted(self, elapsed_s: float) -> bool:
        """True when the whole-turn ceiling is reached."""
        return self.total_s > 0 and elapsed_s >= self.total_s

    def remaining_s(self, elapsed_s: float) -> float:
        """Seconds left before the total ceiling (0 when uncapped/over)."""
        if self.total_s <= 0:
            return 0.0
        return max(0.0, self.total_s - elapsed_s)

    def as_frame(self) -> dict:
        return {"ack_s": self.ack_threshold_s,
                "first_visible_s": self.first_visible_s,
                "total_s": self.total_s}


def load_budget(cfg=None) -> StreamBudget:
    """Build the per-task budget from config (fail-open to config.yaml values)."""
    if cfg is None:
        try:
            from app.core.config_loader import cfg as _cfg
            cfg = _cfg
        except Exception:  # noqa: BLE001
            cfg = None
    perceived = getattr(cfg, "perceived", None)
    llm = getattr(cfg, "llm", None)
    llm_routing = getattr(llm, "routing", None)
    ack = float(getattr(perceived, "ttft_ack_threshold_s", 0.0) or 0.0)
    # first_token_deadline_s lives under llm.routing; chat_stream_budget_s under
    # llm — read both fail-open, with the roadmap defaults.
    first = float(getattr(llm_routing, "first_token_deadline_s", 5.0) or 5.0)
    total = float(getattr(llm, "chat_stream_budget_s", 300.0) or 300.0)
    # An ack that fires after the first-visible deadline is pointless — clamp it.
    if ack and first and ack > first:
        ack = max(0.0, first - 0.5)
    return StreamBudget(ack_threshold_s=ack, first_visible_s=first,
                        total_s=total)


__all__ = ["StreamBudget", "load_budget"]
