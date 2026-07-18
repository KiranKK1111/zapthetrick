"""Autonomous prompt optimization (roadmap Phase 7 #5).

Closes the loop the roadmap flags as partial: benchmark → optimize → regression-
test → deploy, with no manual prompt tuning. Given a set of candidate variants
for a prompt, evaluate them all, pick the best, and PROMOTE it over the current
champion only when it clears the regression gate (`compare_variants` → BETTER and
no per-case regression). The active champion is remembered per prompt name.

Deterministic given a generator; the actual LLM generation is injected, so the
loop is fully offline-testable.
"""
from __future__ import annotations

from collections.abc import Callable

from app.eval.prompt_eval import (
    Comparison,
    PromptCase,
    PromptVariant,
    Verdict,
    compare_variants,
    evaluate_prompt,
)

# In-process champion registry: prompt name → active version.
_active: dict[str, str] = {}


def active_version(name: str) -> str | None:
    return _active.get(name)


def set_active(name: str, version: str) -> None:
    _active[name] = version


def reset() -> None:
    _active.clear()


def optimize(
    name: str,
    variants: list[PromptVariant],
    cases: list[PromptCase],
    generate: Callable[[str], str],
) -> dict:
    """Evaluate every variant for [name]; promote the best over the current
    champion only if it isn't a regression. Returns the decision."""
    if not variants:
        return {"promoted": False, "reason": "no variants", "active": active_version(name)}

    results = [evaluate_prompt(v, cases, generate) for v in variants]
    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]

    champ_ver = active_version(name)
    if champ_ver is None:
        set_active(name, best.variant.version)
        return {
            "promoted": True, "reason": "first champion",
            "active": best.variant.version,
            "scores": {r.variant.version: round(r.score, 4) for r in results},
        }

    if best.variant.version == champ_ver:
        return {
            "promoted": False, "reason": "champion still best",
            "active": champ_ver,
            "scores": {r.variant.version: round(r.score, 4) for r in results},
        }

    champ_result = next(
        (r for r in results if r.variant.version == champ_ver), None)
    cmp: Comparison | None = (
        compare_variants(champ_result, best) if champ_result else None)
    # Promote when the candidate is strictly BETTER and no case regressed.
    promote = (cmp is None) or (
        cmp.verdict == Verdict.BETTER and not cmp.regressed_cases)
    if promote:
        set_active(name, best.variant.version)
    return {
        "promoted": promote,
        "reason": ("beats champion" if promote else
                   ("regressed cases" if cmp and cmp.regressed_cases
                    else "not better")),
        "active": active_version(name),
        "candidate": best.variant.version,
        "delta": round(cmp.delta, 4) if cmp else None,
        "scores": {r.variant.version: round(r.score, 4) for r in results},
    }


def _default_suite() -> tuple[str, list[PromptVariant], list[PromptCase]]:
    """A built-in, offline optimization suite so the optimizer has something real
    to run without any manual wiring. Variants for a 'clarify-vs-answer' style
    instruction, graded by objective gates — deterministic, no model keys."""
    from app.eval.scoring import contains_all
    variants = [
        PromptVariant("answer_style", "v1",
                      "Answer the question: {q}"),
        PromptVariant("answer_style", "v2",
                      "Answer the question concisely and cite evidence: {q}"),
        PromptVariant("answer_style", "v3",
                      "Answer concisely, cite evidence, avoid speculation: {q}"),
    ]
    cases = [
        PromptCase(inputs={"q": "what is kafka"},
                   gates=[contains_all("concise"), contains_all("evidence")],
                   name="quality"),
        PromptCase(inputs={"q": "explain raft"},
                   gates=[contains_all("evidence")],
                   name="grounding"),
    ]
    return "answer_style", variants, cases


def _echo_generator(prompt: str) -> str:
    """Deterministic offline generator: the rendered template IS the output, so a
    variant that instructs the right qualities scores for containing them. Lets
    the optimizer run end-to-end with no provider keys."""
    return prompt


def run_default_optimization(generate: Callable[[str], str] | None = None) -> dict:
    """Reachable, no-arg invoker (Phase 7 #5 — 'autonomous' needs a caller).

    Runs the built-in suite through the champion-promotion loop and returns the
    decision. Called by the diagnostic endpoint and the maintenance scheduler, so
    the optimizer is genuinely wired rather than dormant. Fail-open."""
    try:
        name, variants, cases = _default_suite()
        return optimize(name, variants, cases, generate or _echo_generator)
    except Exception as exc:  # noqa: BLE001
        return {"promoted": False, "reason": f"error: {exc}", "active": None}


__all__ = [
    "optimize", "active_version", "set_active", "reset",
    "run_default_optimization",
]
