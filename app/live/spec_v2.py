"""Speculation as the common case (vNext §4.2, Stage 6 Component F).

Waiting for the interviewer to fully finish before we start thinking wastes the
most valuable ~1 s of every turn. §4.2 makes speculation the DEFAULT: the moment
a partial transcript reads like a plausible complete question (the completeness
gate, `hypothesis.completeness`), we fire the answer speculatively on the fast
tier. When the utterance actually endpoints, we compare the FINAL against the
partial we speculated on — a match ≥ `speculation_endpoint_threshold` (0.92)
means the speculation is valid, so we FLUSH it to the user with ~0 TTFT; a miss
means we hedge a fresh answer inside `hedge_budget_ms`. Per-stage enrichment runs
under a tight deadline (`enrichment_budget_ms`); a stage that would blow the
critical-path budget is DEFERRED to the next turn rather than delaying this one.

This module owns the deterministic decisions of that machine — the trigger gate,
the flush/miss decision, and the enrichment budget. The actual fast-tier draft +
hedge already exist (`predict.py` / the router's speculative path); this decides
WHEN to fire and WHETHER to flush. Self-contained (only `hypothesis`, same
package), pure + fail-open. Flag-gated (`live.speculation_v2`, default OFF).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "speculation_v2", False))
    except Exception:  # noqa: BLE001
        return False


def _endpoint_threshold() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "speculation_endpoint_threshold", 0.92))
    except Exception:  # noqa: BLE001
        return 0.92


def _enrichment_budget_ms() -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.live, "enrichment_budget_ms", 120.0))
    except Exception:  # noqa: BLE001
        return 120.0


# --------------------------------------------------------------------------- #
# 1. Trigger — fire from the first plausible partial
# --------------------------------------------------------------------------- #
@dataclass
class SpecTrigger:
    fire: bool
    completeness: str = "neutral"
    reason: str = ""


def should_speculate(partial: str, *, min_words: int = 4) -> SpecTrigger:
    """Decide whether to fire speculation on this PARTIAL transcript. §4.2 fires
    on the first PLAUSIBLE partial — one the completeness gate does NOT read as
    clearly mid-thought ("incomplete") and that has at least `min_words` (so we
    don't speculate on "so tell"). A miss just wastes a cheap fast-tier draft, so
    the bar is deliberately low. No-op when speculation-v2 is off. Never raises."""
    try:
        if not enabled():
            return SpecTrigger(fire=False, reason="disabled")
        t = (partial or "").strip()
        words = re.findall(r"\w+", t)
        if len(words) < max(1, min_words):
            return SpecTrigger(fire=False, completeness="neutral",
                               reason="too short")
        from app.live.hypothesis import completeness as _completeness
        comp = _completeness(t)
        if comp == "incomplete":
            return SpecTrigger(fire=False, completeness=comp,
                               reason="mid-thought — wait for more")
        # "complete" or "neutral" → plausible enough to speculate.
        return SpecTrigger(fire=True, completeness=comp,
                           reason="plausible question")
    except Exception:  # noqa: BLE001
        return SpecTrigger(fire=False, reason="error")


# --------------------------------------------------------------------------- #
# 2. Flush — reuse the speculative answer when the final matches the partial
# --------------------------------------------------------------------------- #
@dataclass
class FlushDecision:
    flush: bool          # the speculation is valid → reveal it (~0 TTFT)
    similarity: float    # partial↔final similarity (0..1)
    hedge: bool          # a miss → hedge a fresh answer instead


def _similarity(a: str, b: str) -> float:
    ta = re.findall(r"\w+", (a or "").lower())
    tb = re.findall(r"\w+", (b or "").lower())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).ratio()


def flush_decision(speculated_on: str, final: str,
                   *, threshold: float | None = None) -> FlushDecision:
    """Once the utterance endpoints, decide whether the answer we speculated on
    `speculated_on` is still valid for the actual `final`. A similarity ≥ the
    endpoint threshold (0.92) → FLUSH (reveal the speculative answer, ~0 TTFT);
    below it → a MISS, so hedge a fresh answer. Never raises."""
    try:
        thr = _endpoint_threshold() if threshold is None else threshold
        sim = _similarity(speculated_on, final)
        hit = sim >= thr
        return FlushDecision(flush=hit, similarity=round(sim, 3), hedge=not hit)
    except Exception:  # noqa: BLE001
        return FlushDecision(flush=False, similarity=0.0, hedge=True)


# --------------------------------------------------------------------------- #
# 3. Per-stage enrichment budget — keep the critical path tight
# --------------------------------------------------------------------------- #
@dataclass
class EnrichmentBudget:
    """Track cumulative critical-path enrichment time for a turn. A stage whose
    estimate would exceed the budget is DEFERRED (contributes to the next turn),
    never delaying this answer. `budget_ms` defaults from config."""
    budget_ms: float = 0.0
    spent_ms: float = 0.0
    ran: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.budget_ms <= 0:
            self.budget_ms = _enrichment_budget_ms()

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.budget_ms - self.spent_ms)

    def should_run(self, stage: str, est_ms: float) -> bool:
        """Run `stage` only if its estimate fits the remaining critical-path
        budget; otherwise defer it. A non-positive estimate always runs (free)."""
        try:
            if est_ms <= 0 or est_ms <= self.remaining_ms:
                self.spent_ms += max(0.0, est_ms)
                self.ran.append(stage)
                return True
            self.deferred.append(stage)
            return False
        except Exception:  # noqa: BLE001
            return True

    def as_dict(self) -> dict:
        return {"budget_ms": self.budget_ms, "spent_ms": round(self.spent_ms, 1),
                "ran": list(self.ran), "deferred": list(self.deferred)}


__all__ = ["enabled", "SpecTrigger", "should_speculate", "FlushDecision",
           "flush_decision", "EnrichmentBudget"]
