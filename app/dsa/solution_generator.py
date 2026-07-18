"""Stage 4 — Solution generator.

Produces up to three approaches: brute / optimized / optimal. Each
approach carries code + reasoning + (later) complexity proof. The
caller can decide which to feature in the final answer; the
beautifier renders all of them under collapsible headings.

One LLM call. The prompt asks for a strict structured response which
the parser is lenient with — partial outputs still produce at least
one usable approach.
"""
from __future__ import annotations

import logging
import re

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from .types import PatternMatch, ProblemSpec, SolutionApproach
from app.core.prompt import fill

log = logging.getLogger(__name__)

_PROMPT = """You are a coding-interview tutor. Solve the problem below.

Produce up to THREE approaches at three quality levels:
  1. BRUTE — straightforward, correct, often inefficient
  2. OPTIMIZED — a clear improvement (better complexity)
  3. OPTIMAL — the best known approach

For each approach, output EXACTLY this format:

===APPROACH: <BRUTE|OPTIMIZED|OPTIMAL>===
SUMMARY: <one sentence overview>
REASONING:
<step-by-step thought process, why this works, key observations>
CODE:
```{language}
<complete runnable code; function name + signature matching the problem;
no main(), no print statements, no extra commentary>
```

Use the pattern hint when generating the OPTIMAL approach.

PROBLEM:
{statement}

PATTERN HINT: {pattern} ({pattern_rationale})

LANGUAGE: {language}
"""

async def generate(
    problem: ProblemSpec,
    pattern: PatternMatch,
    *,
    language: str = "python",
) -> list[SolutionApproach]:
    """One LLM call → up to 3 [SolutionApproach]s. Robust parser."""
    if not problem.statement.strip():
        return []

    model = cfg.llm.code_model or cfg.llm.model
    prompt = fill(_PROMPT, 
        statement=problem.statement[:6000],
        pattern=pattern.family,
        pattern_rationale=pattern.rationale or "general algorithm",
        language=language,
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=model,
            options={
                "temperature": cfg.temperature.planning,
                "num_predict": max(2500, cfg.llm.max_tokens),
            },
        )
    except LLMError as exc:
        log.warning("solution_generator LLM call failed: %s", exc)
        return []

    return _parse(raw, language=language)

_APPROACH_RE = re.compile(
    r"===\s*APPROACH\s*:\s*(BRUTE|OPTIMIZED|OPTIMAL)\s*===\s*(.*?)(?=(?:===\s*APPROACH\s*:|\Z))",
    re.IGNORECASE | re.DOTALL,
)
_SUMMARY_RE = re.compile(r"SUMMARY\s*:\s*(.+?)(?=\n\s*REASONING\s*:|\n\s*CODE\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+?)(?=\n\s*CODE\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*(.+?)```", re.DOTALL)

def _parse(raw: str, *, language: str) -> list[SolutionApproach]:
    """Pull APPROACH blocks out of the response, tolerant of model drift."""
    if not raw:
        return []
    text = raw.strip()
    out: list[SolutionApproach] = []
    matches = list(_APPROACH_RE.finditer(text))
    if not matches:
        # No section markers — treat the whole response as one OPTIMAL
        # approach. Some smaller models drop the markers; better one
        # usable answer than zero.
        code = _extract_code(text)
        out.append(
            SolutionApproach(
                level="optimal",
                summary=text.splitlines()[0][:200] if text else "",
                reasoning=text[:4000],
                code=code,
                language=language,
            )
        )
        return out

    for m in matches:
        level = m.group(1).lower()
        body = m.group(2)
        summary_m = _SUMMARY_RE.search(body)
        reasoning_m = _REASONING_RE.search(body)
        code = _extract_code(body)
        out.append(
            SolutionApproach(
                level=level,
                summary=(summary_m.group(1).strip() if summary_m else "")[:300],
                reasoning=(reasoning_m.group(1).strip() if reasoning_m else "")[:4000],
                code=code,
                language=language,
            )
        )
    return out

def _extract_code(text: str) -> str:
    """First fenced block, or empty string."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else ""
