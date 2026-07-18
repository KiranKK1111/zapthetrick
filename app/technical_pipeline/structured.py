"""Shared architecture-grade structured output (Phase 7, #19/#40/#41/#88/#142).

Every IT/CS domain pipeline produces the SAME rigorous shape: domain-specific
sections PLUS the four cross-cutting "architecture rigor" sections an opinionated
design assistant should always surface —

  • Assumptions               — what the answer assumes (unstated requirements),
  • Recommended Pattern(s)     — named architecture/design patterns + rationale,
  • Trade-offs                 — pros/cons + at least one alternative considered,
  • Governance & Operability   — security, observability, cost, maintainability.

A `DomainSpec` declares the domain's role, its ordered section headings, and a
verifier checklist. `run_structured_domain` builds the prompt, calls the model,
runs the checklist (+ optional domain lint), shapes the output, and yields the
same event stream as the other pipelines (stage / markdown / verify / done),
so the chat mesh (CoderAgent) renders it unchanged.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.response_arch import Shape, finalize

# Cross-cutting sections + the keywords the verifier looks for to confirm each
# was actually addressed.
ARCHITECTURE_SECTIONS: list[tuple[str, list[str]]] = [
    ("Assumptions",
     ["assumption", "assume", "we assume", "given that", "presume"]),
    ("Recommended Pattern(s)",
     ["pattern", "approach", "style", "architecture", "strategy"]),
    ("Trade-offs",
     ["trade-off", "tradeoff", "trade off", "pros", "cons", "alternative",
      "downside", "versus", " vs "]),
    ("Governance & Operability",
     ["security", "observability", "monitor", "cost", "maintainab",
      "compliance", "scal", "operability", "reliab"]),
]
_ARCH_HEADINGS = [s[0] for s in ARCHITECTURE_SECTIONS]


@dataclass
class DomainSpec:
    domain: str
    role: str                                   # "senior database engineer"
    sections: list[str]                         # domain section headings (order)
    checklist: list[tuple[str, list[str]]]      # (label, keywords) verifier
    shape: Shape = Shape.STEPS
    emit_artifacts: bool = False
    lint: Callable[[str], Iterable[str]] | None = field(default=None)


def _build_prompt(spec: DomainSpec, question: str) -> str:
    headings = list(spec.sections) + _ARCH_HEADINGS
    numbered = "\n".join(f"  {i}. {h}" for i, h in enumerate(headings, 1))
    return (
        f"You are a {spec.role}. Produce a rigorous, opinionated answer to the "
        "question using EXACTLY these section headings as markdown `##` headers, "
        f"in this order:\n{numbered}\n\n"
        "Guidance:\n"
        "- Be concrete: name specific technologies, patterns, and numbers; avoid "
        "vague generalities.\n"
        "- State your ASSUMPTIONS explicitly (the question is usually under-"
        "specified).\n"
        "- Recommend a clear primary approach and name the design/architecture "
        "PATTERN(S) it uses and why.\n"
        "- Give honest TRADE-OFFS with at least one alternative you rejected and "
        "the reason.\n"
        "- Address GOVERNANCE & OPERABILITY: security, observability, cost, and "
        "maintainability.\n"
        "- Where a diagram clarifies the design, include a Mermaid code block.\n\n"
        f"Question:\n{question}\n"
    )


def check_missing(text: str, checklist: list[tuple[str, list[str]]]) -> list[str]:
    """Sections whose keywords don't appear in the answer (verifier misses)."""
    low = (text or "").lower()
    return [label for label, kws in checklist
            if not any(k in low for k in kws)]


async def run_structured_domain(
    question: str, spec: DomainSpec,
) -> AsyncIterator[dict]:
    """Run one domain with architecture-grade structured output + verification."""
    yield {"kind": "stage", "name": "classifier", "data": {"domain": spec.domain}}

    try:
        raw = await llm.complete(
            [{"role": "user", "content": _build_prompt(spec, question)}],
            model=cfg.llm.code_model or cfg.llm.model,
            options={"num_predict": max(2_000, cfg.llm.max_tokens)},
        )
    except LLMError as exc:
        yield {"kind": "done", "data": {"warning": f"llm failed: {exc}"}}
        return

    text = raw or ""
    checklist = list(spec.checklist) + ARCHITECTURE_SECTIONS
    missing = check_missing(text, checklist)
    findings: list[str] = list(spec.lint(text)) if spec.lint else []

    shaped = finalize(text, question=question, shape=spec.shape)
    yield {"kind": "markdown", "text": shaped.text}

    if spec.emit_artifacts and shaped.artifacts:
        yield {
            "kind": "artifacts",
            "items": [
                {"filename": a.filename, "language": a.language,
                 "content": a.content}
                for a in shaped.artifacts
            ],
        }

    errors = [f"missing section: {m}" for m in missing] + findings
    if errors:
        yield {
            "kind": "verify",
            "passed": len(checklist) - len(missing),
            "failed": len(errors),
            "errors": errors,
        }
    yield {"kind": "done", "data": {
        "shape": shaped.shape.value, "domain": spec.domain,
        "missing": missing, "lint": findings,
    }}


__all__ = [
    "DomainSpec", "ARCHITECTURE_SECTIONS", "check_missing",
    "run_structured_domain",
]
