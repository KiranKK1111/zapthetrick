"""Phase C scenarios — live-interview decision & answer-quality.

Deterministic, no-LLM, no-network tests that exercise the REAL implemented
`app/live/*` modules. Each test maps to a numbered scenario. For code paths that
internally call an LLM (e.g. `verify.verify_answer`) we test only the pure,
deterministic helpers (`verify._parse`, `verify.Verdict.ok`, `critique_directive`).

Test names: test_s<NN>_<slug>; each carries a `# Scenario NN: <desc>` comment.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config_loader import cfg
from app.live import (
    decision,
    deliberate,
    evidence,
    guard,
    language,
    latency,
    modes,
    objective,
    plan,
    premise,
    strategy,
    style,
    uncertainty,
    verify,
)


# ── flag helpers (live decisions are cfg.live.* gated) ──────────────────
def _set(**flags):
    saved = {k: getattr(cfg.live, k, False) for k in flags}
    for k, v in flags.items():
        setattr(cfg.live, k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(cfg.live, k, v)


@dataclass
class _Event:
    """Minimal stand-in for a detected event (duck-typed by decide_event)."""
    is_answerable: bool = True
    kind: str = "statement"


# ── Scenario 65 — decision engine ───────────────────────────────────────
def test_s65_decision_engine():
    # Scenario 65: decide_utterance cancels on self-correction, skips feedback
    # and rhetorical asides, and always answers typed input.
    saved = _set(interruption_handling=True, satisfaction_detection=False,
                 rhetorical=False, implicit_question=False)
    try:
        d = decision.decide_utterance("Actually let's talk about RabbitMQ", is_audio=True)
        assert d.action == decision.CANCEL_THEN_ANSWER
        assert d.reason == "interruption"
    finally:
        _restore(saved)

    # A short interviewer reaction is feedback, never a question → SKIP.
    saved = _set(interruption_handling=False, satisfaction_detection=True,
                 rhetorical=False, implicit_question=False)
    try:
        d = decision.decide_utterance("makes sense", is_audio=True)
        assert d.action == decision.SKIP and d.reason == "feedback"
        assert any(f["type"] == "feedback" for f in d.frames)
    finally:
        _restore(saved)

    # A trailing tag-question is rhetorical → SKIP.
    saved = _set(interruption_handling=False, satisfaction_detection=False,
                 rhetorical=True, implicit_question=False)
    try:
        d = decision.decide_utterance("You know what I mean?", is_audio=True)
        assert d.action == decision.SKIP and d.reason == "rhetorical"
    finally:
        _restore(saved)

    # Typed input is always answered (audio rules do not apply).
    d = decision.decide_utterance("Actually leave that", is_audio=False)
    assert d.action == decision.ANSWER


# ── Scenario 68 — answer planning steps ─────────────────────────────────
def test_s68_answer_planning_steps():
    # Scenario 68: make_plan yields an ordered outline for a strategy and
    # as_directive renders it as a one-line prompt directive.
    steps = plan.make_plan("Design a chat app", strategy.DESIGN_SESSION)
    assert steps == ["Requirements & scale", "High-level architecture",
                     "Key components & data flow", "Trade-offs & bottlenecks"]
    directive = plan.as_directive(steps)
    assert directive.startswith("Follow this outline:")
    assert "Trade-offs & bottlenecks" in directive
    # GENERAL has no steps; empty plan → empty directive.
    assert plan.make_plan("anything", strategy.GENERAL) == []
    assert plan.as_directive([]) == ""


# ── Scenario 69 — multi-pass understanding ──────────────────────────────
def test_s69_multi_pass_understanding():
    # Scenario 69: multi_pass runs estimate then refines depth from recent
    # context — a "but why" follow-up escalates a definition to architecture.
    base_obj, base_depth = objective.estimate("What is a mutex?", "technical_concept",
                                              difficulty="trivial")
    assert base_depth == objective.DEFINITION
    obj, depth = objective.multi_pass("What is a mutex?", "technical_concept",
                                      difficulty="trivial",
                                      recent=["but why does it matter"])
    assert obj == base_obj
    assert depth == objective.ARCHITECTURE  # escalated one notch from definition


# ── Scenario 70 — answer strategy selection ─────────────────────────────
def test_s70_answer_strategy_selection():
    # Scenario 70: mode/strategy selection routes question types to the right
    # answer shape (behavioral → STAR story; comparison cue → comparison).
    assert modes.detect_mode("Tell me about a time you had a conflict",
                             "behavioral") == modes.STAR_STORY
    assert modes.directive(modes.STAR_STORY)  # non-empty shaping directive
    assert strategy.select_strategy("behavioral", "") == strategy.STAR
    assert strategy.select_strategy("technical_concept", "",
                                    "difference between TCP and UDP") == strategy.COMPARISON


# ── Scenario 71 — answer verifier question-check ────────────────────────
def test_s71_answer_verifier_question_check():
    # Scenario 71: Verdict.ok is True only when relevance clears the min and
    # hallucination risk stays under the cap.
    good = verify.Verdict(relevance=0.9, hallucination_risk=0.1, issue="")
    assert good.ok is True
    off_topic = verify.Verdict(relevance=0.2, hallucination_risk=0.1, issue="")
    assert off_topic.ok is False
    hallucinated = verify.Verdict(relevance=0.9, hallucination_risk=0.95, issue="")
    assert hallucinated.ok is False
    assert good.to_meta()["verdict"] == "ok"
    assert off_topic.to_meta()["verdict"] == "weak"


# ── Scenario 72 — answer quality scorer ─────────────────────────────────
def test_s72_answer_quality_scorer():
    # Scenario 72: _parse extracts relevance/hallucination from the verifier's
    # JSON (tolerating prose/fences); a low-relevance verdict is weak and
    # critique_directive yields a non-empty regeneration directive.
    v = verify._parse('{"relevance":0.9,"hallucination_risk":0.1,"issue":""}')
    assert v is not None
    assert v.relevance == 0.9 and v.hallucination_risk == 0.1
    assert v.ok is True

    weak = verify._parse('prose ```{"relevance":0.2,"hallucination_risk":0.9,'
                         '"issue":"off topic"}``` trailing')
    assert weak is not None and weak.ok is False
    directive = verify.critique_directive(weak, "What is the CAP theorem?")
    assert isinstance(directive, str) and directive.strip()
    assert "off topic" in directive

    # Non-JSON → None (fail-open, no verdict).
    assert verify._parse("no json here") is None


# ── Scenario 73 — answer lifecycle ──────────────────────────────────────
def test_s73_answer_lifecycle():
    # Scenario 73: the answer lifecycle — post-detection decide_event admits a
    # real question, skips a non-answerable one on the audio path, and
    # admit_answer gates generation (open when the session budget is off).
    saved = _set(ensemble_detection=False)
    try:
        d = decision.decide_event(_Event(is_answerable=True), is_audio=True,
                                  utterance="What is Kafka?")
        assert d.action == decision.ANSWER

        d = decision.decide_event(_Event(is_answerable=False, kind="statement"),
                                  is_audio=True, utterance="I like Kafka.")
        assert d.action == decision.SKIP and d.reason == "statement"

        # Typed non-answerable is still answered (audio-only rule).
        d = decision.decide_event(_Event(is_answerable=False), is_audio=False,
                                  utterance="hi")
        assert d.action == decision.ANSWER
    finally:
        _restore(saved)

    saved = _set(session_budget=False)
    try:
        ok, budget = decision.admit_answer("sid-lifecycle")
        assert ok is True and budget is None
    finally:
        _restore(saved)


# ── Scenario 76 — hallucination prevention ──────────────────────────────
def test_s76_hallucination_prevention():
    # Scenario 76: with no (or stale) supporting evidence, hedge_directive emits
    # a tentative-answer directive; once real evidence is bound it stays silent.
    binding = evidence.EvidenceBinding()
    assert binding.is_thin() is True
    hedge = evidence.hedge_directive(binding)
    assert isinstance(hedge, str) and hedge.strip()

    binding.add("Kafka persists records to disk in append-only segments.", "kb")
    assert binding.is_thin() is False
    assert evidence.hedge_directive(binding) == ""


# ── Scenario 77 — knowledge-gap detection ───────────────────────────────
def test_s77_knowledge_gap_detection():
    # Scenario 77: the guard flags a low-confidence (expert) question — capping
    # to concise + hedged — and passes an easy one through.
    v = guard.assess("expert", threshold=0.5)
    assert v.gap is True and v.hedge is True
    assert v.max_depth == "concise" and v.confidence < 0.5

    ez = guard.assess("trivial", threshold=0.5)
    assert ez.gap is False and ez.hedge is False and ez.confidence >= 0.85
    assert guard.hedge_directive().strip()


# ── Scenario 83 — interview depth estimation ────────────────────────────
def test_s83_interview_depth_estimation():
    # Scenario 83: estimate reads the expected answer depth — a trivial "what
    # is" stays at definition; expert difficulty and internals cues go deeper.
    _, d_trivial = objective.estimate("What is a hash map?", "technical_concept",
                                      difficulty="trivial")
    assert d_trivial == objective.DEFINITION

    _, d_expert = objective.estimate("Explain the scheduler", difficulty="expert")
    assert d_expert == objective.INTERNALS

    _, d_internals = objective.estimate("How does Raft work under the hood?")
    assert d_internals == objective.INTERNALS


# ── Scenario 89 — latency budgeting ─────────────────────────────────────
def test_s89_latency_budgeting():
    # Scenario 89: the fast/deep path is chosen from difficulty, and degraded
    # latency health forces the fast/concise budget.
    hard = latency.select_path("hard")
    assert hard.path == latency.DEEP and hard.depth == "detailed"

    trivial = latency.select_path("trivial")
    assert trivial.path == latency.FAST and trivial.depth == "concise"

    degraded = latency.select_path("expert", latency_degraded=True)
    assert degraded.path == latency.FAST and degraded.depth == "concise"


# ── Scenario 102 — adversarial false premise ────────────────────────────
def test_s102_adversarial_false_premise():
    # Scenario 102: a confirmation-seeking absolute claim is flagged as a false
    # premise so the answer corrects rather than affirms it.
    p = premise.check_premise("Kafka stores data only in memory, right?")
    assert p.false_premise is True
    assert p.confidence >= 0.7
    assert premise.directive(p).strip()

    clean = premise.check_premise("What is Kafka used for?")
    assert clean.false_premise is False
    assert premise.directive(clean) == ""


# ── Scenario 13 — STT confidence lowers answer confidence ───────────────
def test_s13_stt_conf_lowers_answer_conf():
    # Scenario 13: uncertainty.propagate drags a high answer confidence down when
    # STT confidence is low; a high STT signal leaves it unchanged.
    baseline = uncertainty.propagate(0.95)
    assert baseline == 0.95

    low_stt = uncertainty.propagate(0.95, stt_conf=0.2)
    assert low_stt < baseline
    assert low_stt == 0.5 + 0.5 * 0.2  # capped at 0.6

    high_stt = uncertainty.propagate(0.95, stt_conf=0.99)
    assert high_stt == 0.95  # strong STT does not reduce it


# ── Scenario 80 — adaptive answer length ────────────────────────────────
def test_s80_adaptive_answer_length():
    # Scenario 80: deliberation adapts target answer depth — a hard/expert
    # question is capped to a concise answer; a standard behavioral question is
    # not forced concise.
    saved = _set(phase_detection=True, answer_strategy=True, answer_planning=False,
                 knowledge_gap_guard=True, adaptive_latency=True,
                 knowledge_gap_threshold=0.5)
    try:
        hard = deliberate.deliberate("Explain Raft consensus internals",
                                     "technical_concept", "expert")
        assert hard.depth == "concise"
        assert hard.confidence is not None and hard.confidence < 0.5

        easy = deliberate.deliberate("Tell me about a time you led a project",
                                     "behavioral", "standard")
        assert easy.depth != "concise"
    finally:
        _restore(saved)


# ── Scenario 118 — cognitive load ───────────────────────────────────────
def test_s118_cognitive_load():
    # Scenario 118: cognitive-load estimation escalates under rapid-fire pace +
    # pending answers and stays low when the pace is relaxed; the depth
    # directive shortens answers under high load.
    high = style.cognitive_load(questions_per_min=8, pending_answers=3,
                                interviewer_style=style.RAPID_FIRE)
    assert high == style.LOAD_HIGH
    assert style.depth_for_load(high).strip()

    low = style.cognitive_load(questions_per_min=1)
    assert low == style.LOAD_LOW
    assert style.depth_for_load(low) == ""


# ── Bonus coverage — multilingual answer targeting (language module) ────
def test_s70b_language_targeting():
    # Scenario 70 (supporting): detect_language identifies the utterance
    # language for answer targeting, tolerating code-switching punctuation.
    assert language.detect_language("Hello, how are you?") == "en"
    assert language.detect_language("¿cómo estás?") == "es"
