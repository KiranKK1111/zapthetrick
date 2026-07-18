"""Shared dataclasses for the DSA pipeline.

Every stage produces an immutable typed payload that the next stage
consumes — no hidden global state. The Visualizer's output mirrors the
shape Architecture2.md §"Visualizations" specifies so the Flutter
renderer can play frames back via CustomPainter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemSpec:
    """The output of the problem extractor."""
    title: str = ""
    statement: str = ""
    constraints: list[str] = field(default_factory=list)
    input_spec: str = ""
    output_spec: str = ""
    examples: list[dict] = field(default_factory=list)         # [{"input": ..., "output": ..., "note": ...}]
    language_hint: str | None = None


@dataclass
class PatternMatch:
    """One of the spec's 24 pattern families plus confidence."""
    family: str = "general"
    confidence: float = 0.0
    rationale: str = ""                                         # short LLM-or-rule justification


@dataclass
class SolutionApproach:
    """One of {brute, optimized, optimal}. Code + reasoning."""
    level: str = "brute"                                        # brute | optimized | optimal
    summary: str = ""                                           # 1-2 sentence overview
    reasoning: str = ""                                         # full walk-through
    code: str = ""
    language: str = "python"


@dataclass
class ComplexityProof:
    """Time + space complexity with a short proof sketch."""
    time: str = ""                                              # e.g. "O(n log n)"
    space: str = ""
    proof: str = ""                                             # recurrence / counting argument


@dataclass
class TraceFrame:
    """One step of an example trace. Used by the visualizer."""
    note: str = ""                                              # human description of what happened
    state: dict[str, Any] = field(default_factory=dict)         # vars / pointers / data structure snapshot


@dataclass
class VizPayload:
    """Structured visualization data — Architecture2.md §Visualizations.

    Shape examples:
        viz_type='array_with_pointers' → frames=[{array, pointers, highlight, note}, ...]
        viz_type='tree'                → frames=[{nodes, edges, current, visited, note}, ...]
        viz_type='dp_table'            → frames=[{rows, cols, filled_cells, focus, note}, ...]
        viz_type='graph'               → frames=[{nodes, edges, current, queue, note}, ...]

    The Flutter renderer keys on `viz_type` to pick a CustomPainter.
    """
    viz_type: str = "none"
    frames: list[dict[str, Any]] = field(default_factory=list)
    legend: dict[str, str] = field(default_factory=dict)


@dataclass
class VerifyResult:
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: bool = False                                       # True when no runnable language
    repair_attempts: int = 0


@dataclass
class DsaResponse:
    """End-to-end output of the DSA pipeline. The beautifier composes
    `markdown` from the structured fields; routes stream `markdown`
    token-by-token, then emit `viz_payload` + `verify_result` on done."""
    problem: ProblemSpec = field(default_factory=ProblemSpec)
    pattern: PatternMatch = field(default_factory=PatternMatch)
    approaches: list[SolutionApproach] = field(default_factory=list)
    complexity: ComplexityProof = field(default_factory=ComplexityProof)
    examples: list[dict] = field(default_factory=list)
    trace: list[TraceFrame] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    viz: VizPayload = field(default_factory=VizPayload)
    verify: VerifyResult = field(default_factory=VerifyResult)
    markdown: str = ""
    follow_ups: list[str] = field(default_factory=list)
