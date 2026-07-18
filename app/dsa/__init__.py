"""DSA pipeline — Architecture2.md §4.

Specialized sub-pipeline for coding/algorithm questions. The `CoderAgent`
slot in the multi-agent mesh delegates here when `Intent.type == 'coding'`.

Stage chain (each stage is its own module so individual stages can be
swapped for stronger models / cached / disabled via config):

    extractor → classifier → knowledge_rag → solution_generator
    → example_builder → complexity_prover → verifier (+ repair loop)
    → edge_case_auditor → visualizer → beautifier

Entry point: `await dsa.pipeline.solve(question, language=None)` yields
SSE events the route layer can stream to the UI:

    {"kind": "stage",  "name": "classifier", "data": {...}}
    {"kind": "code",   "text": "..."}
    {"kind": "viz",    "frames": [...]}
    {"kind": "verify", "passed": 7, "failed": 1, "errors": [...]}
    {"kind": "done",   "summary": "..."}
"""
from .pipeline import DsaEvent, solve
from .types import (
    ProblemSpec,
    PatternMatch,
    SolutionApproach,
    ComplexityProof,
    TraceFrame,
    VizPayload,
    VerifyResult,
    DsaResponse,
)

__all__ = [
    "solve",
    "DsaEvent",
    "ProblemSpec",
    "PatternMatch",
    "SolutionApproach",
    "ComplexityProof",
    "TraceFrame",
    "VizPayload",
    "VerifyResult",
    "DsaResponse",
]
