"""
Synthetic live-evaluation scenarios (live-conversational-intelligence R27).

Auto-generates annotated `LiveScenario`s over the deterministic live decision
functions to augment the hand-annotated Phase-5 matrix without real recordings.
Every generated scenario is clearly tagged synthetic (name prefix `synth:`),
carries no real PII, and runs with no audio / no provider keys. Dev/CI-only with
no runtime effect on the live path.
"""
from __future__ import annotations

from app.eval.live_scenarios import LiveScenario
from app.live import ensemble, events, phase, strategy, topic_graph

# Templated, PII-free interview material.
_TECH = ["Kafka", "Redis", "Postgres", "Kubernetes", "GraphQL", "gRPC"]
_DEFINITIONS = [f"What is {t}?" for t in _TECH]
_BEHAVIORAL = [
    "Tell me about a time you handled a conflict",
    "Describe a situation where you led a project",
    "Tell me about a time you missed a deadline",
]
_CODING = [
    "Implement a function to reverse a linked list",
    "Write code to find the longest substring without repeating characters",
]
_NON_QUESTIONS = [
    "In our company we use microservices extensively.",
    "Okay, that makes sense.",
    "Good morning, nice to meet you.",
]


def _kind(is_q: bool, qtype: str, raw: str) -> str:
    from app.question_detection.agent import Prediction
    pred = Prediction(is_q, raw if is_q else "", qtype, "")
    if is_q:
        return events.QUESTION
    return events._refine_nonquestion(raw, events._kind_for(pred))  # noqa: SLF001


def generate_scenarios(n: int = 20) -> list[LiveScenario]:
    """Generate up to `n` auto-annotated synthetic scenarios (tagged synthetic)."""
    out: list[LiveScenario] = []

    def add(name, category, fn, check, metric="pass"):
        out.append(LiveScenario(name=f"synth:{name}", category=category,
                                fn=fn, check=check, metric=metric))

    # question_detection — every definition is a question the ensemble keeps.
    for q in _DEFINITIONS:
        add(f"qd:{q[:20]}", "question_detection",
            (lambda: ensemble.decide(agent_is_q=True, agent_conf=0.85,
                                     heuristic_is_q=True, heuristic_conf=0.7).is_question),
            lambda v: v is True)
    # false_answer — non-questions must not be answerable.
    for nq in _NON_QUESTIONS:
        add(f"fa:{nq[:18]}", "false_answer",
            (lambda nq=nq: _kind(False, "technical_concept", nq)),
            lambda v: v != events.QUESTION, metric="false_answer")
    # phase + strategy — behavioral → STAR, coding → coding_flow.
    for b in _BEHAVIORAL:
        add(f"phase:{b[:18]}", "phase",
            (lambda b=b: phase.detect_phase(b, "behavioral")),
            lambda v: v == phase.BEHAVIORAL)
        add(f"strategy:{b[:14]}", "strategy",
            (lambda b=b: strategy.select_strategy("behavioral", phase.BEHAVIORAL, b)),
            lambda v: v == strategy.STAR)
    for c in _CODING:
        add(f"strategy:code:{c[:12]}", "strategy",
            (lambda c=c: strategy.select_strategy("coding", phase.CODING, c)),
            lambda v: v == strategy.CODING_FLOW)
    # topic_switch — unrelated pair drifts, sub-topic does not.
    for i in range(len(_TECH) - 1):
        a, b = _TECH[i].lower(), _TECH[i + 1].lower()
        add(f"ts:{a}->{b}", "topic_switch",
            (lambda a=a, b=b: _drift(a, b)), lambda v: v is True)
        add(f"ts:{a}-sub", "topic_switch",
            (lambda a=a: _drift(a, f"{a} internals")), lambda v: v is False)
    # multi_question
    add("mq:triple", "multi_question",
        (lambda: events.split_questions("What is Kafka, why use it, and how do you scale it?")),
        lambda qs: len(qs) >= 2)

    return out[: max(0, n)] if n else out


def _drift(prev: str, new: str) -> bool:
    g = topic_graph.TopicGraph()
    g.observe(prev)
    return g.observe(new)


def synth_metrics(n: int = 20) -> dict:
    """Run the synthetic matrix → per-category + overall accuracy + false-answer
    rate (same shape as `live_metrics`). Deterministic; no audio/keys."""
    per: dict[str, dict] = {}
    total = passed = 0
    fa_total = fa_fail = 0
    for s in generate_scenarios(n):
        try:
            ok = bool(s.check(s.fn()))
        except Exception:  # noqa: BLE001
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
        "synthetic": True,
        "overall": {"total": total, "passed": passed,
                    "pass_rate": round(passed / total, 3) if total else 0.0},
        "per_category": per,
        "false_answer_rate": round(fa_fail / fa_total, 3) if fa_total else 0.0,
    }


__all__ = ["generate_scenarios", "synth_metrics"]
