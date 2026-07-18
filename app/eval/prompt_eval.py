"""Prompt evaluation framework (roadmap Phase 1 #7).

Benchmark prompt *versions* against objective gates so prompt changes are
measured, not vibes — and a new prompt ships only if it doesn't regress. Reuses
the existing objective `Gate` scorers in `app.eval.scoring`, so evaluation is
deterministic and offline (no keys/models) when driven by a stub generator; the
same API drives a real model-backed generator in production evals.

Shape:
    registry = PromptRegistry()
    registry.register(PromptVariant("answer", "v1", "Answer: {q}"))
    registry.register(PromptVariant("answer", "v2", "Answer concisely: {q}"))
    cases = [PromptCase({"q": "..."}, gates=[contains_all("kafka")])]
    r1 = evaluate_prompt(registry.get("answer", "v1"), cases, generate)
    r2 = evaluate_prompt(registry.get("answer", "v2"), cases, generate)
    verdict = compare_variants(r1, r2)   # BETTER / SAME / WORSE  (regression gate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.eval.scoring import Gate

# A generator renders a prompt to an output. Offline tests inject a deterministic
# stub; production injects a model-backed callable. Kept as a plain callable so
# the framework never imports the router (no keys needed to test it).
Generator = Callable[[str], str]


@dataclass(frozen=True)
class PromptVariant:
    name: str
    version: str
    template: str
    notes: str = ""

    def render(self, inputs: dict) -> str:
        try:
            return self.template.format(**inputs)
        except KeyError as exc:
            raise KeyError(
                f"prompt {self.name}/{self.version} needs input {exc} not supplied"
            ) from None


@dataclass
class PromptCase:
    inputs: dict
    gates: list[Gate]
    name: str = ""


@dataclass
class CaseScore:
    name: str
    score: float          # 0..1 weighted gate pass ratio
    passed_gates: int
    total_gates: int
    failures: list[str] = field(default_factory=list)


@dataclass
class PromptEvalResult:
    variant: PromptVariant
    case_scores: list[CaseScore]

    @property
    def score(self) -> float:
        if not self.case_scores:
            return 0.0
        return sum(c.score for c in self.case_scores) / len(self.case_scores)

    @property
    def pass_rate(self) -> float:
        if not self.case_scores:
            return 0.0
        full = sum(1 for c in self.case_scores if c.score >= 0.999)
        return full / len(self.case_scores)


class PromptRegistry:
    """Named, versioned prompt variants."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], PromptVariant] = {}

    def register(self, variant: PromptVariant) -> None:
        self._by_key[(variant.name, variant.version)] = variant

    def get(self, name: str, version: str) -> PromptVariant:
        return self._by_key[(name, version)]

    def versions(self, name: str) -> list[str]:
        return sorted(v for (n, v) in self._by_key if n == name)


def _score_case(output: str, gates: list[Gate]) -> tuple[float, int, list[str]]:
    if not gates:
        return 1.0, 0, []
    total_w = sum(g.weight for g in gates) or 1.0
    got_w = 0.0
    failures: list[str] = []
    passed = 0
    for g in gates:
        res = g.run(output)
        if res.passed:
            got_w += res.weight
            passed += 1
        else:
            failures.append(f"{res.name}: {res.detail}")
    return got_w / total_w, passed, failures


def evaluate_prompt(
    variant: PromptVariant,
    cases: list[PromptCase],
    generate: Generator,
) -> PromptEvalResult:
    """Render each case through the variant, generate an output, score by gates."""
    scores: list[CaseScore] = []
    for i, case in enumerate(cases):
        prompt = variant.render(case.inputs)
        output = generate(prompt)
        s, passed, failures = _score_case(output, case.gates)
        scores.append(CaseScore(
            name=case.name or f"case[{i}]",
            score=s, passed_gates=passed, total_gates=len(case.gates),
            failures=failures,
        ))
    return PromptEvalResult(variant, scores)


class Verdict:
    BETTER = "better"
    SAME = "same"
    WORSE = "worse"


@dataclass
class Comparison:
    verdict: str
    delta: float           # candidate.score - baseline.score
    baseline_score: float
    candidate_score: float
    regressed_cases: list[str] = field(default_factory=list)


def compare_variants(
    baseline: PromptEvalResult,
    candidate: PromptEvalResult,
    *,
    epsilon: float = 1e-6,
) -> Comparison:
    """Regression gate: deploy the candidate only if it isn't WORSE. Also reports
    any individual case that regressed even when the aggregate improved."""
    delta = candidate.score - baseline.score
    base_by_name = {c.name: c.score for c in baseline.case_scores}
    regressed = [
        c.name for c in candidate.case_scores
        if c.name in base_by_name and c.score < base_by_name[c.name] - epsilon
    ]
    if delta > epsilon:
        verdict = Verdict.BETTER
    elif delta < -epsilon:
        verdict = Verdict.WORSE
    else:
        verdict = Verdict.SAME
    return Comparison(verdict, delta, baseline.score, candidate.score, regressed)


__all__ = [
    "Generator", "PromptVariant", "PromptCase", "CaseScore",
    "PromptEvalResult", "PromptRegistry", "evaluate_prompt",
    "Verdict", "Comparison", "compare_variants",
]
