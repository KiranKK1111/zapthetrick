"""Objective scoring gates for the graded eval harness (P2-2, report_2 §P2-2).

A `Gate` is a named, weighted, *objective* grader over a model's text output:
it returns pass/fail + a short detail, with NO model in the loop. Gates are the
backbone of "measured, not estimated" parity — they check concrete, verifiable
properties (a section is present, a code block exists, a forbidden anti-pattern
is absent, a regex matches) so two models can be compared on the same rubric.

`GradedTask` bundles a prompt with its gates; `grade_output` runs them and
produces a weighted `TaskScore`. The optional LLM judge (see `model_eval`) is a
*separate* axis layered on top — gates first, judge second.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class GateResult:
    name: str
    passed: bool
    weight: float
    detail: str = ""


@dataclass
class Gate:
    """A named, weighted objective check over output text."""
    name: str
    fn: Callable[[str], bool]
    weight: float = 1.0
    describe: str = ""

    def run(self, output: str) -> GateResult:
        try:
            ok = bool(self.fn(output or ""))
            detail = "" if ok else (self.describe or "check failed")
        except Exception as exc:  # noqa: BLE001 — a broken grader fails its gate
            ok = False
            detail = f"grader raised {type(exc).__name__}: {exc}"
        return GateResult(self.name, ok, self.weight, detail)


# ── gate factories (objective, offline, deterministic) ──────────────────────
def contains_all(*terms: str, ci: bool = True, weight: float = 1.0) -> Gate:
    """Pass iff EVERY term appears in the output."""
    def fn(out: str) -> bool:
        hay = out.lower() if ci else out
        return all((t.lower() if ci else t) in hay for t in terms)
    return Gate(f"contains_all({', '.join(terms)})", fn, weight,
                f"expected all of: {', '.join(terms)}")


def contains_any(*terms: str, ci: bool = True, weight: float = 1.0) -> Gate:
    """Pass iff AT LEAST ONE term appears."""
    def fn(out: str) -> bool:
        hay = out.lower() if ci else out
        return any((t.lower() if ci else t) in hay for t in terms)
    return Gate(f"contains_any({', '.join(terms)})", fn, weight,
                f"expected any of: {', '.join(terms)}")


def contains_none(*terms: str, ci: bool = True, weight: float = 1.0) -> Gate:
    """Pass iff NONE of the (forbidden) terms appear — anti-pattern guard."""
    def fn(out: str) -> bool:
        hay = out.lower() if ci else out
        return not any((t.lower() if ci else t) in hay for t in terms)
    return Gate(f"contains_none({', '.join(terms)})", fn, weight,
                f"must avoid: {', '.join(terms)}")


def regex_present(pattern: str, *, flags: int = re.I, weight: float = 1.0) -> Gate:
    """Pass iff the regex matches somewhere in the output."""
    rx = re.compile(pattern, flags)
    return Gate(f"regex({pattern})", lambda out: bool(rx.search(out)), weight,
                f"expected pattern: {pattern}")


def has_code_block(*, lang: str | None = None, weight: float = 1.0) -> Gate:
    """Pass iff a fenced code block exists (optionally of a given language)."""
    if lang:
        rx = re.compile(rf"```[ \t]*{re.escape(lang)}\b", re.I)
        desc = f"expected a ```{lang} code block"
    else:
        rx = re.compile(r"```")
        desc = "expected a fenced code block"
    return Gate(f"code_block({lang or 'any'})", lambda out: bool(rx.search(out)),
                weight, desc)


def has_sections(*headers: str, weight: float = 1.0) -> Gate:
    """Pass iff every header appears as a markdown heading or bold label."""
    def fn(out: str) -> bool:
        low = out.lower()
        for h in headers:
            hl = h.lower()
            # markdown heading, bold, or a "Header:" label
            if not re.search(rf"(^|\n)\s*#+\s*{re.escape(hl)}", low) \
               and f"**{hl}**" not in low \
               and not re.search(rf"(^|\n)\s*{re.escape(hl)}\s*:", low):
                return False
        return True
    return Gate(f"sections({', '.join(headers)})", fn, weight,
                f"expected sections: {', '.join(headers)}")


def min_words(n: int, *, weight: float = 1.0) -> Gate:
    """Pass iff the output has at least `n` whitespace-delimited words."""
    return Gate(f"min_words({n})", lambda out: len(out.split()) >= n, weight,
                f"expected >= {n} words")


def json_parseable(*, weight: float = 1.0) -> Gate:
    """Pass iff the output contains a parseable JSON object/array."""
    import json

    def fn(out: str) -> bool:
        s = out.strip()
        # tolerate a fenced ```json block
        m = re.search(r"```(?:json)?\s*(.+?)```", s, re.S)
        if m:
            s = m.group(1).strip()
        start = min((i for i in (s.find("{"), s.find("[")) if i != -1),
                    default=-1)
        if start == -1:
            return False
        for end in range(len(s), start, -1):
            try:
                json.loads(s[start:end])
                return True
            except Exception:  # noqa: BLE001
                continue
        return False
    return Gate("json_parseable", fn, weight, "expected parseable JSON")


# ── task + scoring ──────────────────────────────────────────────────────────
@dataclass
class GradedTask:
    """A prompt + the objective rubric (gates) that grades any model's answer."""
    id: str
    prompt: str
    gates: list[Gate]
    category: str = "general"
    difficulty: str = "standard"      # easy|standard|hard|expert
    system: str = ""                  # optional system steer
    reference: str = ""               # optional gold answer (for the LLM judge)


@dataclass
class TaskScore:
    task_id: str
    category: str
    difficulty: str
    score: float                      # 0..1 weighted gate pass fraction
    passed: bool                      # score >= threshold
    gate_results: list[GateResult] = field(default_factory=list)
    output_chars: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "score": self.score,
            "passed": self.passed,
            "output_chars": self.output_chars,
            "error": self.error,
            "gates": [
                {"name": g.name, "passed": g.passed, "weight": g.weight,
                 "detail": g.detail}
                for g in self.gate_results
            ],
        }


def grade_output(task: GradedTask, output: str, *,
                 pass_threshold: float = 0.7, error: str = "") -> TaskScore:
    """Run every gate, compute a weight-normalised 0..1 score."""
    results = [g.run(output) for g in task.gates]
    total_w = sum(g.weight for g in task.gates) or 1.0
    got_w = sum(r.weight for r in results if r.passed)
    score = round(got_w / total_w, 3)
    return TaskScore(
        task_id=task.id, category=task.category, difficulty=task.difficulty,
        score=score, passed=(score >= pass_threshold and not error),
        gate_results=results, output_chars=len(output or ""), error=error,
    )


__all__ = [
    "Gate", "GateResult", "GradedTask", "TaskScore",
    "contains_all", "contains_any", "contains_none", "regex_present",
    "has_code_block", "has_sections", "min_words", "json_parseable",
    "grade_output",
]
