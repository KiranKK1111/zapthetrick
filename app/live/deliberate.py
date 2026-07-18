"""
Deliberation aggregator (live-conversational-intelligence Phase 3).

Composes phase detection + answer strategy + planning + knowledge-gap guard +
adaptive latency into a single `Deliberation` for one confirmed question, each
piece gated by its own `cfg.live.*` flag. Produces an answer **directive**
(injected into the SAME generation call — no second blocking LLM call), a target
**depth**, and a surfaced **answer_confidence**. Fully fail-open: with every
flag off it returns an empty deliberation and the answer path is today's.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config_loader import cfg
from app.live import guard as _guard
from app.live import latency as _latency
from app.live import phase as _phase
from app.live import plan as _plan
from app.live import strategy as _strategy

_DEPTH_ORDER = {"concise": 0, "standard": 1, "detailed": 2}


@dataclass
class Deliberation:
    phase: str = ""
    strategy: str = ""
    directive: str = ""
    depth: str | None = None
    confidence: float | None = None


def _most_conservative(a: str | None, b: str | None) -> str | None:
    vals = [d for d in (a, b) if d in _DEPTH_ORDER]
    if not vals:
        return a or b
    return min(vals, key=lambda d: _DEPTH_ORDER[d])


def deliberate(
    question: str,
    qtype: str = "",
    difficulty: str = "standard",
    topic: str = "",
    recent: list[str] | None = None,
    *,
    latency_degraded: bool = False,
) -> Deliberation:
    """Run the flag-gated deliberation for one question. Never raises."""
    out = Deliberation()
    try:
        ph = ""
        if getattr(cfg.live, "phase_detection", False):
            ph = _phase.detect_phase(question, qtype, topic, recent or [])
            out.phase = ph

        directive_parts: list[str] = []

        if getattr(cfg.live, "answer_strategy", False):
            strat = _strategy.select_strategy(qtype, ph, question)
            out.strategy = strat
            scaffold = _strategy.prompt_shaping(strat)
            if scaffold:
                directive_parts.append(scaffold)
            if getattr(cfg.live, "answer_planning", False):
                steps = _plan.make_plan(question, strat)
                pd = _plan.as_directive(steps)
                if pd:
                    directive_parts.append(pd)

        if getattr(cfg.live, "knowledge_gap_guard", False):
            threshold = float(getattr(cfg.live, "knowledge_gap_threshold", 0.5) or 0.5)
            verdict = _guard.assess(difficulty, threshold=threshold)
            out.confidence = verdict.confidence
            out.depth = _most_conservative(out.depth, verdict.max_depth)
            if verdict.hedge:
                directive_parts.append(_guard.hedge_directive())

        if getattr(cfg.live, "adaptive_latency", False):
            choice = _latency.select_path(difficulty, latency_degraded=latency_degraded)
            out.depth = _most_conservative(out.depth, choice.depth)

        out.directive = "\n".join(directive_parts).strip()
        return out
    except Exception:  # noqa: BLE001
        return Deliberation()
