"""Stage 9 — Visualizer.

Turns a [TraceFrame] sequence into a [VizPayload] keyed by pattern
family. The Flutter side selects a CustomPainter from `viz_type`:

    array_with_pointers → two/three-pointer rendering with highlights
    sliding_window      → a window rectangle that slides across an array
    tree                → graph layout with currently-visited node
    graph               → nodes + edges + BFS/DFS queue
    dp_table            → 1-D or 2-D table with filled cells
    linked_list         → nodes joined by arrows
    none                → no visualisation (heuristic fallback)

This stage runs *after* the trace has been built and is purely a data
shaping step — no LLM call. Mapping a [PatternMatch] family to a
viz_type is a small lookup; the frame projection just selects the
relevant keys from the trace's `state` dict so the renderer doesn't
have to guess.
"""
from __future__ import annotations

from typing import Any

from .types import PatternMatch, TraceFrame, VizPayload


_FAMILY_TO_VIZ: dict[str, str] = {
    "arrays/two-pointer": "array_with_pointers",
    "sliding-window": "sliding_window",
    "prefix-sum": "array_with_pointers",
    "binary-search": "array_with_pointers",
    "hashing": "array_with_pointers",
    "monotonic-stack": "array_with_pointers",
    "monotonic-queue": "array_with_pointers",
    "heap/top-k": "array_with_pointers",
    "linked-list": "linked_list",
    "trees": "tree",
    "tries": "tree",
    "graphs": "graph",
    "union-find": "graph",
    "dp": "dp_table",
    "greedy": "array_with_pointers",
    "backtracking": "tree",
    "divide-and-conquer": "tree",
    "bit-manipulation": "none",
    "math/number-theory": "none",
    "strings": "array_with_pointers",
    "segment-tree-bit": "tree",
    "game-theory": "none",
    "randomized": "none",
}


# Per-viz-type set of state keys we forward to the frontend. Anything
# not in this allow-list is dropped so the payload stays tight and
# the renderer doesn't accidentally fight with rogue model output.
_FRAME_KEYS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "array_with_pointers": ("array", "pointers", "highlight", "visited", "note"),
    "sliding_window":      ("array", "window", "sum", "note"),
    "tree":                ("nodes", "edges", "current", "visited", "note"),
    "graph":               ("nodes", "edges", "current", "queue", "stack", "visited", "note"),
    "dp_table":            ("rows", "cols", "filled_cells", "focus", "note"),
    "linked_list":         ("nodes", "current", "note"),
    "none":                (),
}


_LEGENDS: dict[str, dict[str, str]] = {
    "array_with_pointers": {
        "pointers": "Indices being compared / moved",
        "highlight": "Cell being read or written this step",
    },
    "sliding_window": {
        "window": "Active [left, right] window",
        "sum": "Running aggregate over the window",
    },
    "tree": {
        "current": "Node being visited this step",
        "visited": "Nodes already processed",
    },
    "graph": {
        "current": "Node being expanded",
        "queue": "BFS frontier",
        "stack": "DFS path",
    },
    "dp_table": {
        "filled_cells": "Cells computed so far",
        "focus": "Cell being computed this step",
    },
    "linked_list": {
        "current": "Node the cursor is on",
    },
    "none": {},
}


def build(pattern: PatternMatch, trace: list[TraceFrame]) -> VizPayload:
    """Project the trace into a renderer-friendly payload."""
    viz_type = _FAMILY_TO_VIZ.get(pattern.family, "none")
    if viz_type == "none" or not trace:
        return VizPayload(viz_type=viz_type, frames=[], legend=_LEGENDS.get(viz_type, {}))

    allowed = _FRAME_KEYS_BY_TYPE.get(viz_type, ())
    frames: list[dict[str, Any]] = []
    for frame in trace:
        state = dict(frame.state or {})
        projected: dict[str, Any] = {"note": frame.note}
        for key in allowed:
            if key == "note":
                continue
            if key in state:
                projected[key] = state[key]
        # Always include the raw state when no allowed keys matched, so
        # the renderer at least has *something* to draw rather than an
        # empty frame. The Flutter renderer falls back to a JSON dump
        # in that case.
        if len(projected) == 1:
            projected["raw_state"] = state
        frames.append(projected)

    return VizPayload(viz_type=viz_type, frames=frames, legend=_LEGENDS.get(viz_type, {}))
