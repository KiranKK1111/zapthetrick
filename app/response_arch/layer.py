"""The main entry point — `finalize` runs the full shaping pass.

Stages chained here:
    1. pick the shape (or use the caller's hint)
    2. depth-aware truncation (TL;DR / standard / deeper / exhaustive)
    3. markdown enforcement
    4. polish
    5. artifact splitting (when shape == artifact_set)

The function is intentionally synchronous and pure — every input
becomes a deterministic output. Routes call this at the end of an
LLM stream once they have the full text.

TODO (Architecture.md §"Response architecture"):
  - render-aware token weighting during generation (requires
    feedback into the LLM client; out of scope until we have a
    structured-output abstraction).
  - regeneration with a different shape when polish detects the
    chosen shape doesn't fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .artifacts import Artifact, split_artifacts
from .content_router import Shape, pick_shape
from .markdown_enforcer import enforce_markdown
from .polish import polish
from .sanitize import strip_reasoning


# Character caps per depth. The cap only exists to honour an EXPLICIT "TL;DR"
# request — every other depth is generous enough to never chop a normal,
# complete answer (the model's own max_tokens is the real length limit).
# `0` means "no cap". Standard used to be 4_000, which truncated ordinary
# multi-section answers mid-sentence — that's the "(truncated…)" note users saw.
_DEPTH_CAPS: dict[str, int] = {
    "tldr": 10_000,
    "standard": 24_000,
    "deeper": 60_000,
    "exhaustive": 0,
}


@dataclass
class ShapedResponse:
    text: str = ""
    shape: Shape = Shape.PROSE
    depth: str = "standard"
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def finalize(
    text: str,
    *,
    question: str = "",
    shape: Shape | str | None = None,
    depth: str = "standard",
) -> ShapedResponse:
    """Run the shaping pipeline. Returns a [ShapedResponse]."""
    if shape is None:
        shape_enum = pick_shape(question, text or "")
    else:
        shape_enum = shape if isinstance(shape, Shape) else Shape(shape)

    depth_norm = depth if depth in _DEPTH_CAPS else "standard"
    cap = _DEPTH_CAPS[depth_norm]
    warnings: list[str] = []

    # Strip any chain-of-thought that leaked into the answer (inline
    # <think> tags or GPT-OSS harmony channel markers) before anything
    # else, so depth caps + markdown rules act on the real answer only.
    body = strip_reasoning(text or "")
    if cap and len(body) > cap:
        body = body[:cap].rstrip() + (
            "\n\n_(truncated — switch to a deeper view to read the rest)_"
        )
        warnings.append("truncated for depth")

    body = enforce_markdown(body, shape=shape_enum)
    body = polish(body)

    # MermaidDiagramVisualizations.md #1 — recompile every ```mermaid``` fence
    # from its own structure with the deterministic IR generator, so a diagram the
    # MODEL wrote gets the same guarantees as one the generator produced (quoted
    # labels, balanced subgraph/end, legal arrows, a planned layout).
    #
    # Runs AFTER enforce_markdown/polish on purpose: those two deliberately leave
    # fenced blocks untouched, so the diagram this sees is exactly what the client
    # would have rendered. Heavily gated and fail-open (see `diagrams/lane.py`):
    # an unmodelled diagram type or an incomplete lift leaves the fence alone.
    try:
        from app.diagrams.lane import compile_diagrams

        lane = compile_diagrams(body)
        body = lane.text
        if lane.changed:
            warnings.append(
                f"{sum(1 for o in lane.outcomes if o.rewritten)} diagram(s) "
                f"rebuilt from structure")
    except Exception:  # noqa: BLE001 — never let a compiler pass break a turn
        pass

    artifacts: list[Artifact] = []
    if shape_enum == Shape.ARTIFACT_SET:
        artifacts = split_artifacts(body)

    return ShapedResponse(
        text=body,
        shape=shape_enum,
        depth=depth_norm,
        artifacts=artifacts,
        warnings=warnings,
    )


__all__ = ["finalize", "ShapedResponse"]
