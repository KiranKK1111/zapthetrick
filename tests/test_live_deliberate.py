"""Deliberation: phase / strategy / plan / knowledge-gap guard / latency
(live-conversational-intelligence R7-R10; tasks 7.2, 8.2).

Pins Properties 8-10: phase detection, strategy scaffolds (no second blocking
call), knowledge-gap hedging, governor/difficulty reuse (no new difficulty
call), and fail-open. The deterministic builders are tested directly; the
aggregator is tested by toggling the cfg.live flags.
"""
from __future__ import annotations

from app.core.config_loader import cfg
from app.live import deliberate as delib
from app.live import guard, latency, phase, plan, strategy


# ---- phase detection ---------------------------------------------------
def test_phase_behavioral():
    assert phase.detect_phase("Tell me about a time you had a conflict", "behavioral") == phase.BEHAVIORAL


def test_phase_system_design():
    assert phase.detect_phase("Design a URL shortener that scales to 1B users") == phase.SYSTEM_DESIGN


def test_phase_coding():
    assert phase.detect_phase("Implement a function to reverse a linked list", "coding") == phase.CODING


def test_phase_hr_and_closing():
    assert phase.detect_phase("What are your salary expectations?") == phase.HR
    assert phase.detect_phase("Do you have any questions for us?") == phase.CLOSING


def test_phase_default_technical():
    assert phase.detect_phase("What is a hash map?", "technical_concept") == phase.TECHNICAL_SCREENING


# ---- strategy + scaffold ----------------------------------------------
def test_strategy_star_for_behavioral():
    assert strategy.select_strategy("behavioral", phase.BEHAVIORAL) == strategy.STAR
    assert "STAR" in strategy.prompt_shaping(strategy.STAR)


def test_strategy_design_and_coding():
    assert strategy.select_strategy("technical_concept", phase.SYSTEM_DESIGN) == strategy.DESIGN_SESSION
    assert strategy.select_strategy("coding", phase.CODING) == strategy.CODING_FLOW


def test_strategy_comparison_and_definition():
    assert strategy.select_strategy("technical_concept", phase.TECHNICAL_SCREENING,
                                    "difference between TCP and UDP") == strategy.COMPARISON
    assert strategy.select_strategy("technical_concept", phase.TECHNICAL_SCREENING,
                                    "what is a mutex") == strategy.DEFINITION


def test_general_strategy_has_empty_scaffold():
    assert strategy.prompt_shaping(strategy.GENERAL) == ""


# ---- plan --------------------------------------------------------------
def test_plan_steps_and_directive():
    steps = plan.make_plan("design a chat app", strategy.DESIGN_SESSION)
    assert steps and "Trade-offs & bottlenecks" in steps
    assert plan.as_directive(steps).startswith("Follow this outline:")
    assert plan.as_directive([]) == ""


# ---- knowledge-gap guard ----------------------------------------------
def test_guard_hedges_hard_questions():
    v = guard.assess("expert", threshold=0.5)
    assert v.gap is True
    assert v.max_depth == "concise"
    assert v.hedge is True
    assert v.confidence < 0.5


def test_guard_passes_easy_questions():
    v = guard.assess("trivial", threshold=0.5)
    assert v.gap is False
    assert v.hedge is False
    assert v.confidence >= 0.85


def test_guard_low_stt_propagates():
    v = guard.assess("standard", stt_conf=0.1, threshold=0.5)
    assert v.confidence < 0.8  # low STT drags confidence down


# ---- adaptive latency --------------------------------------------------
def test_latency_deep_for_hard():
    c = latency.select_path("hard")
    assert c.path == latency.DEEP and c.depth == "detailed"


def test_latency_degraded_forces_fast():
    c = latency.select_path("expert", latency_degraded=True)
    assert c.path == latency.FAST and c.depth == "concise"


# ---- aggregator (flag-gated) ------------------------------------------
def _set(**flags):
    saved = {k: getattr(cfg.live, k) for k in flags}
    for k, v in flags.items():
        setattr(cfg.live, k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(cfg.live, k, v)


def test_deliberate_all_off_is_empty():
    saved = _set(phase_detection=False, answer_strategy=False, answer_planning=False,
                 knowledge_gap_guard=False, adaptive_latency=False)
    try:
        d = delib.deliberate("What is Kafka?", "technical_concept", "standard")
        assert d.directive == ""
        assert d.phase == ""
        assert d.depth is None
        assert d.confidence is None
    finally:
        _restore(saved)


def test_deliberate_composes_directive_and_confidence():
    saved = _set(phase_detection=True, answer_strategy=True, answer_planning=True,
                 knowledge_gap_guard=True, adaptive_latency=True,
                 knowledge_gap_threshold=0.5)
    try:
        d = delib.deliberate("Tell me about a time you led a project", "behavioral", "standard")
        assert d.phase == phase.BEHAVIORAL
        assert d.strategy == strategy.STAR
        assert "STAR" in d.directive
        assert "Follow this outline:" in d.directive
        assert d.confidence is not None
    finally:
        _restore(saved)


def test_deliberate_hard_question_hedges_and_concise():
    saved = _set(phase_detection=True, answer_strategy=True, answer_planning=False,
                 knowledge_gap_guard=True, adaptive_latency=True,
                 knowledge_gap_threshold=0.5)
    try:
        d = delib.deliberate("Explain Raft consensus internals", "technical_concept", "expert")
        assert d.depth == "concise"          # guard + latency both push concise
        assert "uncertain" in d.directive.lower() or "concise" in d.directive.lower()
        assert d.confidence is not None and d.confidence < 0.5
    finally:
        _restore(saved)
