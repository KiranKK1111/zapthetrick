"""DSA pipeline orchestrator — Architecture2.md §4.

Wires every stage together and yields [DsaEvent]s the route layer can
relay over SSE. Each stage runs in sequence (some are independent and
*could* be parallel, but the LLM rate-limit + sequential reasoning
win out — running them serially keeps the streamed UI legible).

Events emitted (each is a small JSON-serialisable dict):
    {"kind": "stage",    "name": "extractor",        "data": {...}}
    {"kind": "stage",    "name": "classifier",       "data": {...}}
    {"kind": "stage",    "name": "approaches",       "data": {...}}
    {"kind": "stage",    "name": "complexity",       "data": {...}}
    {"kind": "stage",    "name": "examples",         "data": {...}}
    {"kind": "stage",    "name": "edge_cases",       "data": {...}}
    {"kind": "verify",   "passed": 7, "failed": 1, "errors": [...]}
    {"kind": "viz",      "viz_type": "...", "frames": [...], "legend": {}}
    {"kind": "markdown", "text": "<final markdown>"}
    {"kind": "done",     "data": {...}}            ← always last

Repair loop:
    The verifier feeds failed test errors back into the solution
    generator (`prompt += "\n\nThe previous attempt failed these tests:
    ...\nFix the bug."`). Up to `MAX_REPAIRS` attempts.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import AsyncGenerator, TypedDict

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from . import (
    beautifier,
    complexity_prover,
    edge_case_auditor,
    example_builder,
    extractor,
    pattern_classifier,
    solution_generator,
    verifier,
    visualizer,
)
from .types import DsaResponse, SolutionApproach, VerifyResult
from app.core.prompt import fill

log = logging.getLogger(__name__)

MAX_REPAIRS = 2

class DsaEvent(TypedDict, total=False):
    """SSE-friendly event shape. `kind` discriminates."""
    kind: str
    name: str
    data: dict
    text: str
    passed: int
    failed: int
    errors: list[str]
    viz_type: str
    frames: list[dict]
    legend: dict

async def solve(
    raw_problem: str,
    *,
    language: str | None = None,
) -> AsyncGenerator[DsaEvent, None]:
    """Run the full DSA pipeline. Yields events as each stage finishes.

    The caller is expected to be the route layer streaming SSE. The
    final event is always `{"kind": "done", ...}` so consumers can
    stop reading.
    """
    lang = (language or "python").lower()

    # ---- Stage 1: extract -------------------------------------------------
    problem = extractor.extract(raw_problem or "")
    if not problem.statement.strip():
        # Nothing to do — yield a done event so the route closes cleanly.
        yield {
            "kind": "done",
            "data": {"warning": "empty problem; nothing to solve"},
        }
        return
    yield {
        "kind": "stage",
        "name": "extractor",
        "data": {
            "title": problem.title,
            "constraints": problem.constraints,
            "language_hint": problem.language_hint,
        },
    }

    if problem.language_hint and language is None:
        lang = problem.language_hint

    # ---- Stage 2: classify ------------------------------------------------
    pattern = await pattern_classifier.classify(problem)
    yield {
        "kind": "stage",
        "name": "classifier",
        "data": {
            "family": pattern.family,
            "confidence": pattern.confidence,
            "rationale": pattern.rationale,
        },
    }

    # ---- Stage 4: generate approaches ------------------------------------
    approaches = await solution_generator.generate(problem, pattern, language=lang)
    yield {
        "kind": "stage",
        "name": "approaches",
        "data": {"count": len(approaches), "levels": [a.level for a in approaches]},
    }
    optimal = _pick_optimal(approaches)

    # ---- Stage 5: examples + trace ---------------------------------------
    examples, trace = await example_builder.build(problem, optimal)
    yield {
        "kind": "stage",
        "name": "examples",
        "data": {"examples": examples, "trace_steps": len(trace)},
    }

    # ---- Stage 6: complexity ---------------------------------------------
    complexity = await complexity_prover.prove(optimal)
    yield {
        "kind": "stage",
        "name": "complexity",
        "data": {
            "time": complexity.time,
            "space": complexity.space,
            "proof": complexity.proof,
        },
    }

    # ---- Stage 7: verify + repair loop -----------------------------------
    verify_result = await verifier.verify(optimal, examples)
    repair_attempts = 0
    while (
        optimal is not None
        and not verify_result.skipped
        and verify_result.failed > 0
        and repair_attempts < MAX_REPAIRS
    ):
        repair_attempts += 1
        log.info(
            "DSA repair attempt %d/%d — %d failed",
            repair_attempts,
            MAX_REPAIRS,
            verify_result.failed,
        )
        repaired = await _repair(problem, pattern, optimal, verify_result, language=lang)
        if repaired is None or not repaired.code.strip() or repaired.code == optimal.code:
            break
        optimal = repaired
        # Replace the OPTIMAL in `approaches` so the markdown shows the fix.
        approaches = _replace_optimal(approaches, repaired)
        verify_result = await verifier.verify(optimal, examples)

    verify_result.repair_attempts = repair_attempts
    yield {
        "kind": "verify",
        "passed": verify_result.passed,
        "failed": verify_result.failed,
        "errors": verify_result.errors,
    }

    # ---- Stage 8: edge cases ---------------------------------------------
    edge_cases = edge_case_auditor.audit(problem, pattern, optimal)
    yield {
        "kind": "stage",
        "name": "edge_cases",
        "data": {"items": edge_cases},
    }

    # ---- Stage 9: visualizer ---------------------------------------------
    viz = visualizer.build(pattern, trace)
    yield {
        "kind": "viz",
        "viz_type": viz.viz_type,
        "frames": viz.frames,
        "legend": viz.legend,
    }

    # ---- Stage 10: beautifier --------------------------------------------
    response = DsaResponse(
        problem=problem,
        pattern=pattern,
        approaches=approaches,
        complexity=complexity,
        examples=examples,
        trace=trace,
        edge_cases=edge_cases,
        viz=viz,
        verify=verify_result,
        follow_ups=beautifier.make_follow_ups(problem, pattern),
    )
    response.markdown = beautifier.compose(response)
    yield {"kind": "markdown", "text": response.markdown}

    yield {
        "kind": "done",
        "data": {
            "language": lang,
            "family": pattern.family,
            "verify": {
                "passed": verify_result.passed,
                "failed": verify_result.failed,
                "skipped": verify_result.skipped,
                "repair_attempts": repair_attempts,
            },
        },
    }

# ---- helpers ------------------------------------------------------------
def _pick_optimal(approaches: list[SolutionApproach]) -> SolutionApproach | None:
    """Prefer optimal > optimized > brute > anything."""
    if not approaches:
        return None
    order = {"optimal": 0, "optimized": 1, "brute": 2}
    return sorted(approaches, key=lambda a: order.get(a.level, 99))[0]

def _replace_optimal(
    approaches: list[SolutionApproach], replacement: SolutionApproach
) -> list[SolutionApproach]:
    out: list[SolutionApproach] = []
    replaced = False
    for a in approaches:
        if not replaced and a.level == replacement.level:
            out.append(replacement)
            replaced = True
        else:
            out.append(a)
    if not replaced:
        out.append(replacement)
    return out

_REPAIR_PROMPT = """The previous attempt at this problem failed verification. Fix the bug and reply with corrected code only — same function signature, same language.

PROBLEM:
{statement}

PREVIOUS CODE:
```{language}
{code}
```

TEST FAILURES:
{errors}

Reply with ONLY a fenced code block in the same language. No prose, no headings.
"""

async def _repair(
    problem,
    pattern,
    optimal: SolutionApproach,
    verify_result: VerifyResult,
    *,
    language: str,
) -> SolutionApproach | None:
    """Ask the code model to fix the failing implementation. One round.

    Returns None on LLM failure or no parseable fix; the orchestrator
    falls back to the previous attempt's verify counts in that case.
    """
    model = cfg.llm.code_model or cfg.llm.model
    prompt = fill(_REPAIR_PROMPT, 
        statement=problem.statement[:3000],
        language=optimal.language or language,
        code=optimal.code[:3000],
        errors="\n".join(f"- {e}" for e in verify_result.errors[:6]) or "- (unknown)",
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=model,
            options={"temperature": 0.1, "num_predict": max(2000, cfg.llm.max_tokens)},
        )
    except LLMError as exc:
        log.warning("repair LLM call failed: %s", exc)
        return None
    import re as _re

    m = _re.search(r"```(?:\w+)?\s*(.+?)```", raw or "", _re.DOTALL)
    code = (m.group(1) if m else (raw or "")).strip()
    if not code:
        return None
    return SolutionApproach(
        level=optimal.level,
        summary=optimal.summary,
        reasoning=(optimal.reasoning + "\n\n_Repaired after a failed verification._").strip(),
        code=code,
        language=optimal.language or language,
    )

def response_to_dict(resp: DsaResponse) -> dict:
    """Convenience for routes that want the full structured payload
    instead of streaming events."""
    return {
        "problem": asdict(resp.problem),
        "pattern": asdict(resp.pattern),
        "approaches": [asdict(a) for a in resp.approaches],
        "complexity": asdict(resp.complexity),
        "examples": resp.examples,
        "trace": [asdict(t) for t in resp.trace],
        "edge_cases": resp.edge_cases,
        "viz": asdict(resp.viz),
        "verify": asdict(resp.verify),
        "markdown": resp.markdown,
        "follow_ups": resp.follow_ups,
    }

__all__ = ["solve", "DsaEvent", "response_to_dict", "MAX_REPAIRS"]

# Quiet flake8 by referencing `asyncio` — pipeline schedules subroutines
# but never explicitly uses asyncio primitives outside the called
# stages. Keeping the import to avoid surprising future edits.
_ = asyncio
