"""Pick a response "shape" from the question + draft text.

Eight shapes from Architecture.md §"Shape templates":

    prose         — long-form explanation
    table         — when the answer is naturally tabular
    comparison    — A vs B trade-off
    steps         — ordered procedural list
    code          — code-dominated answer
    artifact_set  — multiple files (Dockerfile + compose + k8s, etc.)
    diagram       — answer should render as a Mermaid / custom viz
    trade_off     — opinionated pro/con

Choice is heuristic — look for marker words / structure in the
question and draft. The downstream beautifier renders accordingly.
"""
from __future__ import annotations

import re
from enum import Enum


class Shape(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    COMPARISON = "comparison"
    STEPS = "steps"
    CODE = "code"
    ARTIFACT_SET = "artifact_set"
    DIAGRAM = "diagram"
    TRADE_OFF = "trade_off"


# Cue word → shape. First match wins.
_CUE_PATTERNS: list[tuple[Shape, re.Pattern]] = [
    (Shape.COMPARISON, re.compile(r"\b(vs\.?|versus|compare|difference between)\b", re.I)),
    (Shape.TRADE_OFF, re.compile(r"\b(pros and cons|trade[- ]?offs?|when to use)\b", re.I)),
    (Shape.STEPS, re.compile(r"\b(steps?|how to|procedure|walk me through)\b", re.I)),
    (Shape.DIAGRAM, re.compile(r"\b(diagram|architecture|system design|flow chart|er diagram|sequence diagram)\b", re.I)),
    (Shape.TABLE, re.compile(r"\b(table|matrix|grid|columns?\s+of)\b", re.I)),
    (Shape.ARTIFACT_SET, re.compile(r"\b(dockerfile|docker[- ]compose|k8s|kubernetes|terraform|ci/cd|pipeline)\b", re.I)),
    (Shape.CODE, re.compile(r"\b(implement|code|function|class|algorithm|write a)\b", re.I)),
]


def pick_shape(question: str, draft: str = "") -> Shape:
    """Choose a shape from question + (optional) draft."""
    haystack = f"{question}\n{draft[:2000]}"
    for shape, pat in _CUE_PATTERNS:
        if pat.search(haystack):
            return shape
    # Multi-fenced-code-block draft → artifact_set even without a cue.
    if draft.count("```") >= 4:
        return Shape.ARTIFACT_SET
    return Shape.PROSE


__all__ = ["Shape", "pick_shape"]
