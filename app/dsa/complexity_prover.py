"""Stage 6 — Complexity prover.

Derives T(n), S(n), and a short proof sketch for the chosen approach.
Two strategies:

  1. **Regex hints from the code** — when the code clearly has a single
     for-loop with `binary_search` keywords, we can short-circuit and
     skip the LLM. Saves a round-trip on the easy cases.
  2. **LLM fallback** for everything else. Strict structured output
     so the parser stays simple.

The output's `proof` field renders inline in the beautifier so the
user sees *why*, not just the Big-O.
"""
from __future__ import annotations

import logging
import re

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from .types import ComplexityProof, SolutionApproach
from app.core.prompt import fill

log = logging.getLogger(__name__)

_HEURISTIC_HINTS: list[tuple[re.Pattern, ComplexityProof]] = [
    # Binary search variants
    (
        re.compile(r"\b(bisect|binary[\s_]?search|while\s+lo\s*<=?\s*hi)", re.I),
        ComplexityProof(
            time="O(log n)",
            space="O(1)",
            proof="Halves the search range each iteration → log₂(n) steps.",
        ),
    ),
    # Sorting
    (
        re.compile(r"\.sort\s*\(|sorted\s*\(", re.I),
        ComplexityProof(
            time="O(n log n)",
            space="O(n) for sort buffer",
            proof="Sort dominates the work; everything else is at most linear.",
        ),
    ),
]

_PROMPT = """You are an algorithms tutor. Derive the time and space complexity for the code below.

Reply in this exact format on three lines, no prose:

TIME: <O(...) expression>
SPACE: <O(...) expression>
PROOF: <one-paragraph proof: recurrence, counting argument, or amortised analysis>

CODE:
```{language}
{code}
```
"""

async def prove(approach: SolutionApproach | None) -> ComplexityProof:
    if approach is None or not approach.code.strip():
        return ComplexityProof()

    # 1. Cheap heuristics.
    for regex, hint in _HEURISTIC_HINTS:
        if regex.search(approach.code):
            return hint

    # 2. LLM fallback.
    model = cfg.llm.classifier_model or cfg.llm.code_model or cfg.llm.model
    try:
        raw = await llm.complete(
            [{"role": "user", "content": fill(_PROMPT, code=approach.code, language=approach.language)}],
            model=model,
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.verdict},
        )
    except LLMError as exc:
        log.warning("complexity_prover LLM call failed: %s", exc)
        return ComplexityProof()

    return _parse(raw)

def _parse(raw: str) -> ComplexityProof:
    if not raw:
        return ComplexityProof()
    time = _line(raw, "TIME")
    space = _line(raw, "SPACE")
    proof = _line(raw, "PROOF", multiline=True)
    return ComplexityProof(time=time, space=space, proof=proof)

def _line(text: str, label: str, *, multiline: bool = False) -> str:
    pattern = rf"{label}\s*:\s*(.+?)" + (r"\Z" if multiline else r"\n")
    m = re.search(pattern, text + "\n", re.IGNORECASE | (re.DOTALL if multiline else 0))
    return m.group(1).strip() if m else ""
