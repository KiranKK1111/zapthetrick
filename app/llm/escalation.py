"""Confidence-based escalation (intelligent-model-routing R5).

`run_with_escalation` serves a faster/cheaper step first and advances along an
Escalation_Chain to a stronger model ONLY when the produced answer is
low-confidence or fails verification (R5.1/R5.2); it stops as soon as the
threshold is met (R5.3). It reuses the caller-supplied confidence/verification
(the `evaluation-and-reliability` Aggregate_Confidence and/or `verified_answer`
verdict) rather than adding a new verification system (R5.4). Disabled or a
single-step chain → exactly one generation, as today (R5.5, Property 5).

The model call + the confidence evaluation are injected so this stays pure and
testable, and so the live path plugs in `route_request` + the existing
verification without this module re-implementing either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class EscalationResult:
    answer: str
    confidence: float
    steps_used: int
    escalated: bool
    trace: list[dict] = field(default_factory=list)


async def run_with_escalation(
    chain: list,
    generate: Callable[[object], Awaitable[str]],
    confidence_fn: Callable[[str], float],
    *,
    threshold: float = 0.6,
    enabled: bool = True,
) -> EscalationResult:
    """Walk `chain` (ordered faster→stronger). For each step generate an answer,
    score it with `confidence_fn`, and stop once the score ≥ `threshold`. The
    best-scoring answer is returned. Disabled / single step → one generation."""
    steps = list(chain or [])
    if not steps:
        return EscalationResult("", 0.0, 0, False)

    if not enabled:
        steps = steps[:1]

    best_answer = ""
    best_conf = -1.0
    trace: list[dict] = []
    used = 0
    for i, step in enumerate(steps):
        used += 1
        try:
            answer = await generate(step)
        except Exception:  # noqa: BLE001 — a failed step escalates (R5.2)
            trace.append({"step": i, "error": True})
            continue
        try:
            conf = float(confidence_fn(answer))
        except Exception:  # noqa: BLE001
            conf = 0.0
        trace.append({"step": i, "confidence": round(conf, 3)})
        if conf > best_conf:
            best_conf, best_answer = conf, answer
        if conf >= threshold:
            break        # threshold met → do NOT escalate further (R5.3)

    return EscalationResult(
        answer=best_answer,
        confidence=max(0.0, best_conf),
        steps_used=used,
        escalated=used > 1,
        trace=trace,
    )


def escalation_chain(difficulty: str) -> list[str]:
    """A default faster→stronger difficulty chain for a turn. Trivial/standard
    start fast and can climb; hard/expert start strong (short chain)."""
    d = (difficulty or "standard").lower()
    if d in ("hard", "expert"):
        return [d]
    if d == "trivial":
        return ["trivial", "standard", "hard"]
    return ["standard", "hard"]


__all__ = ["EscalationResult", "run_with_escalation", "escalation_chain"]
