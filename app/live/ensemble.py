"""
Ensemble question detection (live-conversational-intelligence R12).

Extends `question_detection.fusion` to a multi-signal decision that blends the
rule heuristic, the LLM agent, and (when audio is present) prosody into one
question/not-question call with a blended confidence. The LLM agent carries the
most weight, so the ensemble never DROPS an agent-confirmed question (no new
false negatives); its value is reducing false positives (answering an
explanation) and surfacing a detection confidence. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Weights: rule (heuristic) / agent (LLM) / prosody. The agent dominates.
_W_RULE = 0.25
_W_AGENT = 0.55
_W_PROSODY = 0.20


@dataclass
class EnsembleDecision:
    is_question: bool
    score: float
    components: dict = field(default_factory=dict)


def decide(
    *,
    agent_is_q: bool,
    agent_conf: float = 0.85,
    heuristic_is_q: bool = False,
    heuristic_conf: float = 0.5,
    prosody_score: float | None = None,
    threshold: float = 0.5,
) -> EnsembleDecision:
    """Blend the signals into one decision. `*_is_q` are booleans; `*_conf` are
    that signal's confidence in [0,1]; `prosody_score` is the acoustic
    question-ness in [0,1] (None when no audio). Never raises."""
    try:
        def _sub(is_q: bool, conf: float) -> float:
            c = max(0.0, min(1.0, conf))
            return c if is_q else (1.0 - c)

        rule = _sub(heuristic_is_q, heuristic_conf)
        agent = _sub(agent_is_q, agent_conf)
        if prosody_score is None:
            # Re-normalize over the two available signals.
            total = _W_RULE + _W_AGENT
            score = (_W_RULE * rule + _W_AGENT * agent) / (total or 1.0)
            comps = {"rule": rule, "agent": agent}
        else:
            pros = max(0.0, min(1.0, prosody_score))
            score = _W_RULE * rule + _W_AGENT * agent + _W_PROSODY * pros
            comps = {"rule": rule, "agent": agent, "prosody": pros}
        score = max(0.0, min(1.0, score))
        return EnsembleDecision(is_question=score >= threshold, score=score, components=comps)
    except Exception:  # noqa: BLE001
        # Fail open to the agent's own decision.
        return EnsembleDecision(is_question=bool(agent_is_q),
                                score=1.0 if agent_is_q else 0.0, components={})
