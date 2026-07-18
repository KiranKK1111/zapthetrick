"""Formal ResponsePlan — the pre-token contract (roadmap Phase 6 #5, #11, #12).

Before the first token lands, a turn already knows a lot about the *shape* of
what is coming: which logical sections the answer will have, whether an artifact
(file / diagram / table) is expected, and roughly how heavy it is. Emitting that
as a formal ``plan`` frame lets the client paint a real skeleton (sized sections,
an artifact placeholder) instead of a spinner — the "first meaningful paint".

Three roadmap items land here, all deterministic + fail-open (no LLM call):

* **#5  Response plan / first meaningful paint** — :func:`build_response_plan`
  enumerates the upcoming ``sections`` before the first token.
* **#12 Predictive artifact pre-build** — the plan predicts likely artifacts
  (kind + tentative filename) so the client can pre-allocate a card and the
  backend can pre-warm a slot.
* **#11 Progressive refinement** — :meth:`ResponsePlan.outline` yields the
  section skeleton (outline→expand), and ``refinable`` marks whether an
  outline-first pass is worthwhile.

A bad input degrades to a minimal single-section plan, never an exception.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .content_router import Shape, pick_shape

# Section blueprints per shape. Each entry: (section_id, human title, kind).
# `kind` lets the client size / style the skeleton block.
_SECTIONS: dict[Shape, list[tuple[str, str, str]]] = {
    Shape.PROSE: [("answer", "Answer", "prose")],
    Shape.STEPS: [
        ("overview", "Overview", "prose"),
        ("steps", "Steps", "list"),
        ("notes", "Notes", "prose"),
    ],
    Shape.COMPARISON: [
        ("summary", "Summary", "prose"),
        ("comparison", "Comparison", "table"),
        ("recommendation", "Recommendation", "prose"),
    ],
    Shape.TRADE_OFF: [
        ("overview", "Overview", "prose"),
        ("pros", "Pros", "list"),
        ("cons", "Cons", "list"),
        ("verdict", "Verdict", "prose"),
    ],
    Shape.CODE: [
        ("explanation", "Explanation", "prose"),
        ("code", "Code", "code"),
        ("usage", "Usage", "prose"),
    ],
    Shape.ARTIFACT_SET: [
        ("overview", "Overview", "prose"),
        ("files", "Files", "artifact"),
    ],
    Shape.DIAGRAM: [
        ("overview", "Overview", "prose"),
        ("diagram", "Diagram", "diagram"),
        ("explanation", "Explanation", "prose"),
    ],
    Shape.TABLE: [
        ("summary", "Summary", "prose"),
        ("table", "Table", "table"),
    ],
}

# Shapes whose answer benefits from an outline-first (progressive) render.
_REFINABLE = frozenset(
    {Shape.STEPS, Shape.COMPARISON, Shape.TRADE_OFF, Shape.ARTIFACT_SET}
)

# Cue → predicted artifact filename for common deployment / config asks (#12).
_ARTIFACT_HINTS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bdocker[- ]?compose\b", re.I), "docker-compose.yml", "yaml"),
    (re.compile(r"\bdockerfile\b", re.I), "Dockerfile", "dockerfile"),
    (re.compile(r"\b(k8s|kubernetes|manifest|deployment\.yaml)\b", re.I),
     "deployment.yaml", "yaml"),
    (re.compile(r"\bterraform\b", re.I), "main.tf", "hcl"),
    (re.compile(r"\b(ci/cd|github actions|workflow)\b", re.I),
     "ci.yml", "yaml"),
    (re.compile(r"\brequirements\.txt\b", re.I), "requirements.txt", "text"),
    (re.compile(r"\bpackage\.json\b", re.I), "package.json", "json"),
]


@dataclass
class PlannedSection:
    id: str
    title: str
    kind: str  # prose | list | code | table | diagram | artifact

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind}


@dataclass
class ResponsePlan:
    """The `response.plan` contract emitted before the first token."""

    shape: Shape = Shape.PROSE
    depth: str = "standard"
    sections: list[PlannedSection] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)  # predicted (#12)
    refinable: bool = False

    def outline(self) -> list[str]:
        """Section titles only — the outline-first skeleton (#11)."""
        return [s.title for s in self.sections]

    def as_frame(self) -> dict:
        """Payload for the SSE ``plan`` event (additive, fail-open)."""
        return {
            "shape": self.shape.value,
            "depth": self.depth,
            "sections": [s.as_dict() for s in self.sections],
            "artifacts": list(self.artifacts),
            "refinable": self.refinable,
        }


def predict_artifacts(question: str, shape: Shape) -> list[dict]:
    """Predict the artifacts a turn is likely to produce (#12).

    Pure heuristic over the question + shape. Returns ``{kind, filename,
    language, predicted: True}`` dicts a client can render as placeholders and
    the backend can pre-warm. Empty when nothing is confidently predictable.
    """
    q = question or ""
    out: list[dict] = []
    seen: set[str] = set()
    for pat, fname, lang in _ARTIFACT_HINTS:
        if pat.search(q) and fname not in seen:
            seen.add(fname)
            out.append({"kind": "code", "filename": fname,
                        "language": lang, "predicted": True})
    # A generic code turn predicts one code artifact even without a filename cue.
    if not out and shape == Shape.CODE:
        out.append({"kind": "code", "predicted": True})
    if shape == Shape.DIAGRAM and not out:
        out.append({"kind": "diagram", "predicted": True})
    return out


def build_response_plan(
    question: str,
    *,
    intent: str | None = None,
    shape: Shape | str | None = None,
    depth: str = "standard",
    draft_hint: str = "",
) -> ResponsePlan:
    """Assemble a :class:`ResponsePlan` for a turn (before its first token).

    ``shape`` may be given (from an intent profile); otherwise it is picked from
    the question. Never raises — any failure returns a minimal one-section plan.
    """
    try:
        if shape is None:
            shape_enum = pick_shape(question or "", draft_hint or "")
        else:
            shape_enum = shape if isinstance(shape, Shape) else Shape(shape)
    except Exception:  # noqa: BLE001
        shape_enum = Shape.PROSE

    try:
        blueprint = _SECTIONS.get(shape_enum, _SECTIONS[Shape.PROSE])
        sections = [PlannedSection(i, t, k) for (i, t, k) in blueprint]
        artifacts = predict_artifacts(question or "", shape_enum)
        return ResponsePlan(
            shape=shape_enum,
            depth=depth if depth else "standard",
            sections=sections,
            artifacts=artifacts,
            refinable=shape_enum in _REFINABLE,
        )
    except Exception:  # noqa: BLE001
        return ResponsePlan(
            shape=Shape.PROSE, depth="standard",
            sections=[PlannedSection("answer", "Answer", "prose")],
        )


__all__ = [
    "PlannedSection",
    "ResponsePlan",
    "build_response_plan",
    "predict_artifacts",
]
