"""Response architecture layer — Architecture.md §"Response architecture".

Wraps every model output through a uniform shaping pipeline:

    1. content_router   pick a "shape" (prose, table, comparison, steps,
                        code, artifact-set, diagram, trade-off)
    2. layer            generate against the shape's template
    3. markdown_enforcer guarantee well-formed markdown
    4. polish            small textual cleanups (whitespace, headings,
                        consistent bullets)
    5. artifacts         when the shape is artifact-set, split into
                        individual files for the multi-artifact card

Public surface:
    finalize(text, shape=None, depth='standard') -> ShapedResponse

This layer is intentionally cheap — heuristic-first, no LLM call.
The LLM does the creative work; this layer just makes the output
look the way the user expects.

NOTE: The deepest commitments in the architecture doc — fully
template-aware regeneration, render-aware token weighting — are
left as TODOs and called out inline. The scaffold below is
runnable end-to-end and the public surface is stable.
"""
from .layer import ShapedResponse, finalize
from .content_router import Shape, pick_shape
from .markdown_enforcer import enforce_markdown
from .polish import polish
from .artifacts import Artifact, split_artifacts
# Phase 6 backend streaming/rendering surfaces (all fail-open, additive).
from .plan import ResponsePlan, build_response_plan, predict_artifacts
from .stream_shape import StreamMode, stream_mode_for
from .blocks import Block, BlockAssembler
from .budget import StreamBudget, load_budget
from .analytics import StreamAnalytics
from .priority import PriorityBuffer, frame_priority, sort_frames
from .orchestrator import ResponseOrchestrator


__all__ = [
    "Shape",
    "ShapedResponse",
    "Artifact",
    "finalize",
    "pick_shape",
    "enforce_markdown",
    "polish",
    "split_artifacts",
    # Phase 6
    "ResponsePlan",
    "build_response_plan",
    "predict_artifacts",
    "StreamMode",
    "stream_mode_for",
    "Block",
    "BlockAssembler",
    "StreamBudget",
    "load_budget",
    "StreamAnalytics",
    "PriorityBuffer",
    "frame_priority",
    "sort_frames",
    "ResponseOrchestrator",
]
