"""Intelligent diagram gate (roadmap Phase 5 #21).

A diagram is worth rendering only when it carries structure that prose conveys
poorly — a flow, a sequence of interactions, a component/architecture layout, a
hierarchy, an entity-relationship, a state machine. For a single fact, a short
answer, or linear narrative, a diagram is noise. Existing code can PRODUCE
mermaid (`artifacts/discipline.py`, `documents/generators.py`); what was missing
is the DECISION of whether one improves understanding.

`should_diagram(text, *, request="")` scores structural signals in the answer
(and, lightly, the request) and returns a `DiagramDecision(render, kind, score,
reasons)`. Deterministic, dependency-free, fail-open (error → don't render).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Diagram kinds we can suggest (map onto mermaid diagram types downstream).
FLOWCHART = "flowchart"
SEQUENCE = "sequence"
CLASS = "class"
STATE = "state"
ER = "er"
COMPARISON = "comparison"
NONE = "none"

# Signal → (kind, weight). Multiple hits accumulate.
_SIGNALS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\b(step\s*\d|then\b.*\bnext\b|first,.*then|→|-->|proceeds? to|"
                r"pipeline|workflow|flow\b)", re.I), FLOWCHART, 0.4),
    (re.compile(r"\b(request|response|client|server|calls?|sends?|returns?|"
                r"handshake|round[- ]?trip|ack\b|then replies)\b", re.I), SEQUENCE, 0.3),
    (re.compile(r"\b(architecture|component|service|module|layer|subsystem|"
                r"microservice|topology|deployment)\b", re.I), FLOWCHART, 0.3),
    (re.compile(r"\b(class|inherits?|extends?|implements?|has-a|is-a|"
                r"composition|aggregation|inherit)\b", re.I), CLASS, 0.35),
    (re.compile(r"\b(state|transition|idle\b|pending\b|when .* becomes|"
                r"moves? from .* to)\b", re.I), STATE, 0.3),
    (re.compile(r"\b(entity|relationship|table|foreign key|primary key|"
                r"one-to-many|many-to-many|schema)\b", re.I), ER, 0.35),
    (re.compile(r"\b(vs\.?|versus|compared to|trade[- ]?offs?|pros and cons|"
                r"on the other hand|whereas)\b", re.I), COMPARISON, 0.3),
]

# Explicit user asks — a strong yes regardless of structure heuristics.
_EXPLICIT = re.compile(r"\b(diagram|flowchart|flow chart|draw|visuali[sz]e|"
                       r"sequence diagram|sketch (?:it|the)|show me a graph)\b", re.I)
# Content shapes that a diagram would NOT help.
_MIN_CHARS = 200                 # very short answers don't need a diagram
_ARROW = re.compile(r"(-->|→|->|=>)")


@dataclass
class DiagramDecision:
    render: bool
    kind: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"render": self.render, "kind": self.kind,
                "score": round(self.score, 3), "reasons": self.reasons}


def _count_entities(text: str) -> int:
    """Rough count of distinct Capitalised/technical tokens — a diagram needs
    several interacting things to be worth it."""
    toks = set(re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", text))
    return len(toks)


def should_diagram(text: str, *, request: str = "",
                   threshold: float = 0.55) -> DiagramDecision:
    """Decide whether a diagram improves understanding of `text`.

    Returns render=True only when structural signals clear `threshold` (or the
    user explicitly asked). Short/linear/single-fact answers → render=False.
    """
    try:
        t = text or ""
        req = request or ""
        reasons: list[str] = []

        if _EXPLICIT.search(req) or _EXPLICIT.search(t):
            return DiagramDecision(True, FLOWCHART, 1.0, ["explicitly requested"])

        if len(t.strip()) < _MIN_CHARS:
            return DiagramDecision(False, NONE, 0.0,
                                   [f"answer too short ({len(t.strip())}c) — prose is clearer"])

        kind_scores: dict[str, float] = {}
        for pat, kind, weight in _SIGNALS:
            hits = len(pat.findall(t))
            if hits:
                add = weight * min(hits, 3) / 3.0 + weight * 0.5
                kind_scores[kind] = kind_scores.get(kind, 0.0) + add
                reasons.append(f"{kind} signal x{hits}")

        entities = _count_entities(t)
        arrows = len(_ARROW.findall(t))
        multi_entity_bonus = 0.25 if entities >= 4 else (0.1 if entities >= 3 else 0.0)
        arrow_bonus = min(0.2, 0.05 * arrows)

        if not kind_scores:
            return DiagramDecision(False, NONE, 0.0,
                                   ["no structural signal — a diagram would add noise"])

        best_kind = max(kind_scores, key=lambda k: kind_scores[k])
        score = min(1.0, kind_scores[best_kind] + multi_entity_bonus + arrow_bonus)
        if entities >= 3:
            reasons.append(f"{entities} distinct entities")
        if arrows:
            reasons.append(f"{arrows} explicit arrows/flows")

        # A diagram needs BOTH a structural signal and enough interacting parts;
        # a lone keyword in linear prose is not enough.
        render = score >= threshold and (entities >= 3 or arrows >= 1
                                         or len(kind_scores) >= 2)
        if not render:
            reasons.append(f"score {score:.2f} < {threshold} or too few interacting parts")
        return DiagramDecision(render, best_kind if render else NONE, score, reasons)
    except Exception:  # noqa: BLE001 — never break a turn; default to no diagram
        return DiagramDecision(False, NONE, 0.0, ["diagram gate error — no diagram"])


__all__ = [
    "DiagramDecision", "should_diagram",
    "FLOWCHART", "SEQUENCE", "CLASS", "STATE", "ER", "COMPARISON", "NONE",
]
