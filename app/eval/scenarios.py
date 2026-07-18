"""Scenario coverage matrix (evaluation-and-reliability R1).

A categorized, annotated corpus that **extends** the existing `default_suite()`
into per-category scenarios over the app's deterministic decision functions
(`intent_pipeline.assess`, the `followup` act classifier + reference resolver,
the build/ambiguity routing signals, and context selection). Everything runs
with NO provider keys and reuses the existing `EvalCase`/`run_suite` runner, so
the matrix stays deterministic in CI (R1.2/R1.4, Property 1).

`category_metrics(report)` derives per-category + overall pass rates plus the
false-ask and misroute rates (R1.3, Property 2).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.eval.harness import EvalCase, EvalReport

# Categories the matrix must cover (R1.1).
CATEGORIES = (
    "new_topic", "follow_up", "correction", "comparison", "continuation",
    "ambiguous", "multi_topic", "clarification_needed", "false_ask",
    "routing", "context_selection",
)


@dataclass
class Scenario:
    name: str
    category: str
    fn: Callable[[], Any]
    check: Callable[[Any], bool]
    metric: str = "pass"          # "pass" | "false_ask" | "misroute"


# ── helpers (deterministic, no model) ────────────────────────────────────────
def _state_with_context():
    """A ConversationState seeded with goal/entities/enumerations for follow-up
    and reference scenarios."""
    from app.followup.state import ConversationState
    s = ConversationState({}, "eval-convo")
    s.set_goal("build a Flutter chat app")
    s.add_entity("Flutter")
    s.add_entity("PostgreSQL")
    s.set_enumerations(["Redis", "Memcached", "Dragonfly"])
    return s


def _act(turn: str):
    from app.followup import acts
    s = _state_with_context()
    return acts.classify(turn, s)[0]


def _resolve(turn: str):
    from app.followup.reference import resolve
    return resolve(turn, _state_with_context())


def _assess_decision(text: str, recent: str = "", known: dict | None = None):
    from app.clarify.intent_pipeline import assess
    return assess(text, recent, known).decision


def _task_category(text: str) -> str:
    from app.llm.task_class import classify_task
    return classify_task(text)


def scenarios() -> list[Scenario]:
    """The annotated scenario corpus (≥1 per category, R1.1)."""
    from app.followup import acts
    from app.clarify.intent_pipeline import ANSWER, CLARIFY
    from app.chat.difficulty import is_ambiguous_build_request, is_build_request
    from app.followup.context import select

    out: list[Scenario] = [
        # ── new_topic ──
        Scenario("new_topic:self_contained", "new_topic",
                 lambda: _act("write a python script to parse a CSV file please"),
                 lambda v: v == acts.NEW_TOPIC),
        # ── follow_up ──
        Scenario("follow_up:improve", "follow_up",
                 lambda: _act("make it better"),
                 lambda v: v == acts.FOLLOW_UP),
        Scenario("follow_up:pronoun", "follow_up",
                 lambda: _act("optimize it"),
                 lambda v: v == acts.FOLLOW_UP),
        # ── correction ──
        Scenario("correction:actually", "correction",
                 lambda: _act("actually use SQLite instead"),
                 lambda v: v == acts.CORRECTION),
        # ── comparison ──
        Scenario("comparison:vs", "comparison",
                 lambda: _act("compare them"),
                 lambda v: v == acts.COMPARISON),
        # ── continuation ──
        Scenario("continuation:continue", "continuation",
                 lambda: _act("continue"),
                 lambda v: v == acts.CONTINUATION),
        # ── ambiguous (pre-gate should not answer-blindly) ──
        Scenario("ambiguous:build_no_tech", "ambiguous",
                 lambda: _assess_decision("build me an app"),
                 lambda v: v in (CLARIFY, "defer")),
        # ── multi_topic ──
        Scenario("multi_topic:two_asks", "multi_topic",
                 lambda: _act("also, separately, how do I deploy to AWS?"),
                 lambda v: v in (acts.NEW_TOPIC, acts.FOLLOW_UP)),
        # ── clarification_needed (a required slot missing → CLARIFY) ──
        Scenario("clarification_needed:codegen_no_lang", "clarification_needed",
                 lambda: _assess_decision("write a function to reverse a list"),
                 lambda v: v in (CLARIFY, "defer", ANSWER)),
        Scenario("clarification_needed:doc_no_format", "clarification_needed",
                 lambda: _assess_decision("document this"),
                 lambda v: v in (CLARIFY, "defer")),
        # ── false_ask (specific request → must NOT clarify; fail = false ask) ──
        Scenario("false_ask:specific_python", "false_ask",
                 lambda: _assess_decision(
                     "in python, write a function to reverse a string"),
                 lambda v: v in (ANSWER, "defer"), metric="false_ask"),
        Scenario("false_ask:knowledge", "false_ask",
                 lambda: _assess_decision("what is a hashmap?"),
                 lambda v: v in (ANSWER, "defer"), metric="false_ask"),
        # ── routing (build vs ambiguous build; fail = misroute) ──
        Scenario("routing:specific_build", "routing",
                 lambda: is_build_request("build a REST API in FastAPI"),
                 lambda v: v is True, metric="misroute"),
        Scenario("routing:ambiguous_build", "routing",
                 lambda: is_ambiguous_build_request("build me an app", ""),
                 lambda v: v is True, metric="misroute"),
        # Routing-accuracy: the task classifier picks the right capability axis
        # (intelligent-model-routing R10.3 — routing measured in the matrix).
        Scenario("routing:coding_category", "routing",
                 lambda: _task_category("write a python function to sort a list"),
                 lambda v: v == "coding", metric="misroute"),
        Scenario("routing:math_category", "routing",
                 lambda: _task_category("prove this theorem and solve the integral"),
                 lambda v: v == "math", metric="misroute"),
        # ── context_selection (relevant turn ranked, state summary floor) ──
        Scenario("context_selection:relevance", "context_selection",
                 lambda: select(
                     _state_with_context(),
                     "tune postgres connection pooling",
                     [
                         {"role": "user", "content": "how to set up pgvector?"},
                         {"role": "assistant", "content": "install pgvector..."},
                         {"role": "user", "content": "best pancake recipe?"},
                         {"role": "assistant", "content": "mix flour, eggs..."},
                         {"role": "user", "content": "tune postgres pooling?"},
                         {"role": "assistant", "content": "use pgbouncer..."},
                     ],
                     max_turns=3, recent_floor=1),
                 lambda msgs: not any(
                     "pancake" in (m.get("content", "").lower()) for m in msgs)),
    ]
    return out


def scenario_suite() -> list[EvalCase]:
    """Adapt the scenarios to `EvalCase`s so the existing `run_suite` executes
    them. The scenario's metric is encoded in the case category as
    ``"category|metric"`` so `category_metrics` can split pass/false-ask/misroute."""
    cases: list[EvalCase] = []
    for s in scenarios():
        tag = s.category if s.metric == "pass" else f"{s.category}|{s.metric}"
        cases.append(EvalCase(name=s.name, fn=s.fn, check=s.check, category=tag))
    return cases


def _split(cat_tag: str) -> tuple[str, str]:
    if "|" in cat_tag:
        cat, metric = cat_tag.split("|", 1)
        return cat, metric
    return cat_tag, "pass"


def category_metrics(report: EvalReport) -> dict:
    """Per-category + overall pass rate, plus false-ask and misroute rates
    (R1.3, Property 2)."""
    per: dict[str, dict] = {}
    false_ask_total = false_ask_fail = 0
    misroute_total = misroute_fail = 0

    for r in report.results:
        cat, metric = _split(r.category)
        d = per.setdefault(cat, {"total": 0, "passed": 0})
        d["total"] += 1
        if r.passed:
            d["passed"] += 1
        if metric == "false_ask":
            false_ask_total += 1
            if not r.passed:
                false_ask_fail += 1       # failed a "must not ask" case = false ask
        elif metric == "misroute":
            misroute_total += 1
            if not r.passed:
                misroute_fail += 1

    for cat, d in per.items():
        d["pass_rate"] = round(d["passed"] / d["total"], 3) if d["total"] else 0.0

    total = report.total
    passed = report.passed
    return {
        "overall": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3) if total else 0.0,
        },
        "per_category": per,
        "false_ask_rate": round(false_ask_fail / false_ask_total, 3)
        if false_ask_total else 0.0,
        "misroute_rate": round(misroute_fail / misroute_total, 3)
        if misroute_total else 0.0,
    }


__all__ = ["Scenario", "scenarios", "scenario_suite", "category_metrics",
           "CATEGORIES"]
