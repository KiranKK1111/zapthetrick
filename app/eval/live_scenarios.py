"""
Offline live-evaluation scenario matrix (live-conversational-intelligence R15).

An annotated, deterministic corpus over the Live module's **decision functions**
(`events.split_questions`/`split_boundary` + kind derivation, `topic_graph`
drift, `ensemble.decide`, `phase.detect_phase`, `strategy.select_strategy`,
`satisfaction.classify_feedback`). It runs with **no audio and no provider
keys** — every fn here is deterministic — so it is safe in CI and has zero
runtime effect on the live path.

`live_metrics()` derives per-category + overall accuracy plus the false-answer
rate (answering a non-question), mirroring the `evaluation-and-reliability`
metrics shape so the same baseline/regression machinery applies.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.live import ensemble, events, phase, satisfaction, strategy, topic_graph

CATEGORIES = (
    "question_detection", "boundary", "multi_question", "topic_switch",
    "phase", "strategy", "satisfaction", "false_answer",
)


@dataclass
class LiveScenario:
    name: str
    category: str
    fn: Callable[[], Any]
    check: Callable[[Any], bool]
    metric: str = "pass"          # "pass" | "false_answer"


# ── deterministic helpers (no LLM, no audio) ────────────────────────────
def _kind(is_q: bool, qtype: str, raw: str) -> str:
    """Mirror `type_utterance`'s deterministic kind derivation without the LLM."""
    from app.question_detection.agent import Prediction
    pred = Prediction(is_q, raw if is_q else "", qtype, "")
    if is_q:
        return events.QUESTION
    return events._refine_nonquestion(raw, events._kind_for(pred))  # noqa: SLF001


def _drift(prev: str, new: str) -> bool:
    g = topic_graph.TopicGraph()
    g.observe(prev)
    return g.observe(new)


def scenarios() -> list[LiveScenario]:
    """The annotated live decision corpus (>=1 per category)."""
    return [
        # ── question_detection (ensemble) ──
        LiveScenario("qd:agent_question_kept", "question_detection",
                     lambda: ensemble.decide(agent_is_q=True, agent_conf=0.85,
                                             heuristic_is_q=True, heuristic_conf=0.7,
                                             prosody_score=0.6).is_question,
                     lambda v: v is True),
        LiveScenario("qd:agent_no_fp", "question_detection",
                     lambda: ensemble.decide(agent_is_q=False, agent_conf=0.85,
                                             heuristic_is_q=True, heuristic_conf=0.7,
                                             prosody_score=0.2).is_question,
                     lambda v: v is False),
        LiveScenario("qd:confident_question_not_dropped", "question_detection",
                     lambda: ensemble.decide(agent_is_q=True, agent_conf=0.85,
                                             heuristic_is_q=False, heuristic_conf=0.2,
                                             prosody_score=0.1).is_question,
                     lambda v: v is True),
        # ── boundary ──
        LiveScenario("boundary:context_then_question", "boundary",
                     lambda: events.split_boundary(
                         "We use Kafka. Ordering matters. How do you handle duplicates?",
                         "How do you handle duplicates?"),
                     lambda r: len(r[0]) == 2 and r[1].lower().startswith("how")),
        # ── multi_question ──
        LiveScenario("mq:three_questions", "multi_question",
                     lambda: events.split_questions(
                         "What is Kafka, why is it used, and how would you scale it?"),
                     lambda qs: len(qs) >= 2),
        LiveScenario("mq:single_question", "multi_question",
                     lambda: events.split_questions("How do partitions work?"),
                     lambda qs: len(qs) == 1),
        # ── topic_switch ──
        LiveScenario("ts:kafka_to_redis_drift", "topic_switch",
                     lambda: _drift("kafka", "redis"),
                     lambda v: v is True),
        LiveScenario("ts:kafka_subtopic_no_drift", "topic_switch",
                     lambda: _drift("kafka", "kafka partitions"),
                     lambda v: v is False),
        # ── phase ──
        LiveScenario("phase:behavioral", "phase",
                     lambda: phase.detect_phase("Tell me about a time you had a conflict",
                                                "behavioral"),
                     lambda v: v == phase.BEHAVIORAL),
        LiveScenario("phase:system_design", "phase",
                     lambda: phase.detect_phase("Design a URL shortener that scales to 1B"),
                     lambda v: v == phase.SYSTEM_DESIGN),
        LiveScenario("phase:coding", "phase",
                     lambda: phase.detect_phase("Implement a function to reverse a list",
                                                "coding"),
                     lambda v: v == phase.CODING),
        # ── strategy ──
        LiveScenario("strategy:star", "strategy",
                     lambda: strategy.select_strategy("behavioral", phase.BEHAVIORAL),
                     lambda v: v == strategy.STAR),
        LiveScenario("strategy:design", "strategy",
                     lambda: strategy.select_strategy("technical_concept", phase.SYSTEM_DESIGN),
                     lambda v: v == strategy.DESIGN_SESSION),
        LiveScenario("strategy:comparison", "strategy",
                     lambda: strategy.select_strategy("technical_concept",
                                                      phase.TECHNICAL_SCREENING,
                                                      "difference between TCP and UDP"),
                     lambda v: v == strategy.COMPARISON),
        # ── satisfaction ──
        LiveScenario("sat:closed", "satisfaction",
                     lambda: satisfaction.classify_feedback("Great, makes sense"),
                     lambda v: v == satisfaction.CLOSED),
        LiveScenario("sat:open", "satisfaction",
                     lambda: satisfaction.classify_feedback("not quite, think deeper"),
                     lambda v: v == satisfaction.OPEN),
        # ── false_answer (a non-question must NOT be answerable) ──
        LiveScenario("fa:explanation", "false_answer",
                     lambda: _kind(False, "technical_concept",
                                   "In our company we use Kafka extensively."),
                     lambda v: v != events.QUESTION, metric="false_answer"),
        LiveScenario("fa:greeting", "false_answer",
                     lambda: _kind(False, "smalltalk", "Good morning, nice to meet you."),
                     lambda v: v != events.QUESTION, metric="false_answer"),
        LiveScenario("fa:acknowledgement", "false_answer",
                     lambda: _kind(False, "smalltalk", "Okay, got it."),
                     lambda v: v != events.QUESTION, metric="false_answer"),
    ]


def live_metrics() -> dict:
    """Run the matrix and return per-category + overall accuracy + false-answer
    rate. Deterministic; no audio, no provider keys; no runtime effect."""
    per: dict[str, dict] = {}
    total = passed = 0
    fa_total = fa_fail = 0

    for s in scenarios():
        ok = False
        try:
            ok = bool(s.check(s.fn()))
        except Exception:  # noqa: BLE001 — a broken scenario counts as a failure
            ok = False
        total += 1
        passed += 1 if ok else 0
        d = per.setdefault(s.category, {"total": 0, "passed": 0})
        d["total"] += 1
        d["passed"] += 1 if ok else 0
        if s.metric == "false_answer":
            fa_total += 1
            if not ok:
                fa_fail += 1

    for d in per.values():
        d["pass_rate"] = round(d["passed"] / d["total"], 3) if d["total"] else 0.0

    return {
        "overall": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3) if total else 0.0,
        },
        "per_category": per,
        "false_answer_rate": round(fa_fail / fa_total, 3) if fa_total else 0.0,
    }


__all__ = ["LiveScenario", "scenarios", "live_metrics", "CATEGORIES"]
