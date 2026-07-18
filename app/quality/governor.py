"""Request governor + pipeline selection (evaluation-and-reliability R5).

`select_pipeline(difficulty, budgets, health) -> Pipeline` chooses a **fast** vs
**deep** pipeline from the EXISTING difficulty classification
(`classify_difficulty` → trivial|standard|hard|expert) and per-request budgets.
A trivial request early-exits to the minimal pipeline (input → model → output),
skipping retrieval / planning / validation (R5.2); a complex request gets the
deep pipeline (R5.3).

The governor **consumes** the perceived-speed router/observatory health signals;
it never re-implements routing or latency measurement (R5.4). Flags / data
absent → today's pipeline (R5.5, Property 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

FAST = "fast"
DEEP = "deep"

# Stages each pipeline traverses (consumed by the caller to skip stages).
_FAST_STAGES = ("input", "model", "output")
_DEEP_STAGES = ("input", "retrieval", "planning", "model", "validation", "output")


@dataclass
class Budgets:
    latency_ms: int = 0          # 0 = no explicit latency budget
    quality: str = "balanced"    # "fast" | "balanced" | "thorough"
    cost: str = "normal"         # "low" | "normal" | "high"


@dataclass
class Pipeline:
    kind: str                    # "fast" | "deep"
    stages: tuple[str, ...]
    reasons: list[str] = field(default_factory=list)

    @property
    def is_fast(self) -> bool:
        return self.kind == FAST

    def skips(self, stage: str) -> bool:
        return stage not in self.stages


def _deep(reasons: list[str]) -> Pipeline:
    return Pipeline(DEEP, _DEEP_STAGES, reasons)


def _fast(reasons: list[str]) -> Pipeline:
    return Pipeline(FAST, _FAST_STAGES, reasons)


def select_pipeline(difficulty: str | None, budgets: Budgets | None = None,
                    health=None) -> Pipeline:
    """Pick fast vs deep. Trivial → fast/early-exit; hard/expert → deep;
    standard → fast unless a thorough quality budget is requested. Fail-open to
    DEEP (today's full pipeline) on any error / unknown input (R5.5)."""
    try:
        return _select(difficulty, budgets or Budgets(), health)
    except Exception:  # noqa: BLE001
        return _deep(["governor error — default deep pipeline"])


def _select(difficulty: str | None, budgets: Budgets, health) -> Pipeline:
    level = (difficulty or "").lower()
    # The governor honours the Fast/Balanced/Thorough LEVER natively through
    # `budgets.quality` (P5 #26). The user's reasoning mode maps onto that string
    # via `llm.reasoning_mode.quality_budget()`; the caller builds the Budgets so
    # the quality package stays free of a cross-package import to `llm`
    # (import-boundary guardrail).
    quality = budgets.quality

    # A thorough quality budget always gets the deep pipeline.
    if quality == "thorough":
        return _deep([f"quality budget=thorough (difficulty={level or 'n/a'})"])

    if level == "trivial":
        return _fast(["trivial difficulty → fast pipeline, early exit"])

    if level in ("hard", "expert"):
        return _deep([f"{level} difficulty → deep pipeline"])

    if level == "standard":
        # A tight latency budget or an explicitly fast quality budget → fast.
        if quality == "fast" or (0 < budgets.latency_ms <= 1500):
            return _fast(["standard difficulty under a fast/latency budget"])
        return _deep(["standard difficulty → deep pipeline"])

    # Unknown / missing difficulty → today's full pipeline (fail-open, R5.5).
    return _deep(["unknown difficulty → default deep pipeline"])


__all__ = ["Budgets", "Pipeline", "select_pipeline", "FAST", "DEEP"]
