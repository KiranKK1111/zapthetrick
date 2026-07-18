"""Stage 5 — Example builder.

Crafts 2–3 input cases (small / edge / large) and per-step traces.
Uses the problem's existing examples when present; generates synthetic
ones otherwise. The trace frames double as input for the visualizer
(stage 8).

One LLM call. The traces are small structured JSON — the parser is
defensive against models that drift into prose.
"""
from __future__ import annotations

import json
import logging
import re

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

from .types import ProblemSpec, SolutionApproach, TraceFrame
from app.core.prompt import fill

log = logging.getLogger(__name__)

_PROMPT = """You build worked examples for coding-interview problems.

Given the problem + an optimal solution, generate THREE test cases:
  1. SMALL — a tiny case showing the algorithm's basic behaviour
  2. EDGE — boundary input (empty / single-element / max-size / duplicates / negative)
  3. LARGE — non-trivial but still hand-traceable

For each case, also produce a 3–8 step trace showing the algorithm's
state at each iteration. The trace becomes the visualisation animation.

Reply with a single JSON object, no prose:

{{
  "examples": [
    {{ "label": "small",  "input": "...", "expected_output": "...", "note": "..." }},
    {{ "label": "edge",   "input": "...", "expected_output": "...", "note": "..." }},
    {{ "label": "large",  "input": "...", "expected_output": "...", "note": "..." }}
  ],
  "trace": [
    {{ "note": "Initialize pointers", "state": {{ "...": "..." }} }},
    {{ "note": "...",                  "state": {{ "...": "..." }} }}
  ]
}}

PROBLEM:
{statement}

OPTIMAL CODE:
```{language}
{code}
```
"""

async def build(
    problem: ProblemSpec,
    optimal: SolutionApproach | None,
) -> tuple[list[dict], list[TraceFrame]]:
    """Return (examples, trace_frames). Empty lists on LLM failure —
    the beautifier renders a "no trace available" note in that case.
    """
    if optimal is None or not optimal.code.strip():
        return list(problem.examples), []

    model = cfg.llm.code_model or cfg.llm.model
    prompt = fill(_PROMPT, 
        statement=problem.statement[:3000],
        code=optimal.code[:3000],
        language=optimal.language,
    )
    try:
        raw = await llm.chat_json(
            [{"role": "user", "content": prompt}],
            model=model,
        )
    except LLMError as exc:
        log.warning("example_builder LLM call failed: %s", exc)
        return list(problem.examples), []

    return _parse(raw, fallback_examples=problem.examples)

def _parse(raw: str, *, fallback_examples: list[dict]) -> tuple[list[dict], list[TraceFrame]]:
    if not raw:
        return list(fallback_examples), []
    cleaned = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return list(fallback_examples), []
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return list(fallback_examples), []

    examples_raw = obj.get("examples", []) or []
    if not isinstance(examples_raw, list):
        examples_raw = []
    examples = [
        {
            "label": str(e.get("label", "")),
            "input": str(e.get("input", "")),
            "expected_output": str(e.get("expected_output", "")),
            "note": str(e.get("note", "")),
        }
        for e in examples_raw
        if isinstance(e, dict)
    ] or list(fallback_examples)

    trace_raw = obj.get("trace", []) or []
    if not isinstance(trace_raw, list):
        trace_raw = []
    trace = [
        TraceFrame(
            note=str(f.get("note", "")),
            state=f.get("state", {}) if isinstance(f.get("state"), dict) else {},
        )
        for f in trace_raw
        if isinstance(f, dict)
    ]

    return examples, trace
