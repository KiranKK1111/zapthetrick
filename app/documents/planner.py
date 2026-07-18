"""Document Planner + Blueprint — Phase 2 of the Document Generation roadmap.

DocuementGeneration.md's #1 planner idea: don't generate the document directly —
first build a PLAN (a blueprint of sections at a target depth), then generate.
This module supplies the deterministic planning stage:

  * ``detect_document_goal(text)`` — the user's real goal (executive report /
    interview notes / technical design / proposal / research / how-to / general),
    from the request, not a format keyword (Step 1 / second-doc Goal Detector).
  * ``detect_depth(text)`` — quick / medium / detailed (adaptive depth, #24).
  * ``plan_blueprint(goal, depth)`` — the section set for that goal, scaled by
    depth (dynamic section planning, #3): a Blueprint the generator/validator
    consume so the document is structured on purpose.

Deterministic + fail-open: an unrecognized request → GENERAL goal, MEDIUM depth,
no imposed template (the answer's own structure stands). Nothing here forces a
document to exist — that's triage's call (see Phase 0). This only shapes one
that's already going to be produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class DocGoal(str, Enum):
    EXECUTIVE_REPORT = "executive_report"
    INTERVIEW_NOTES = "interview_notes"
    TECHNICAL_DESIGN = "technical_design"
    PROPOSAL = "proposal"
    RESEARCH = "research"
    HOW_TO = "how_to"
    MEETING_MINUTES = "meeting_minutes"
    GENERAL = "general"


class Depth(str, Enum):
    QUICK = "quick"        # ~2–5 pages: required sections only
    MEDIUM = "medium"      # ~8–20 pages: required + core optional
    DETAILED = "detailed"  # ~30–60 pages: everything


@dataclass
class PlannedSection:
    title: str
    required: bool = True
    description: str = ""


@dataclass
class Blueprint:
    goal: DocGoal
    depth: Depth
    sections: list[PlannedSection] = field(default_factory=list)
    est_pages: int = 0

    def titles(self) -> list[str]:
        return [s.title for s in self.sections]

    def as_dict(self) -> dict:
        return {
            "goal": self.goal.value, "depth": self.depth.value,
            "est_pages": self.est_pages,
            "sections": [{"title": s.title, "required": s.required,
                          "description": s.description} for s in self.sections],
        }


# ── goal detection (ordered: most specific first) ───────────────────────────
_GOAL_PATTERNS: list[tuple[DocGoal, re.Pattern]] = [
    (DocGoal.INTERVIEW_NOTES, re.compile(
        r"\binterview\s+(notes?|prep|preparation|questions?|summary|feedback)\b"
        r"|\bprepare\b[^.?!\n]{0,20}\binterview\b", re.I)),
    (DocGoal.MEETING_MINUTES, re.compile(
        r"\bmeeting\s+(minutes?|notes?|summary)\b|\bminutes\s+of\s+the\b", re.I)),
    (DocGoal.TECHNICAL_DESIGN, re.compile(
        r"\b(design\s+doc(?:ument)?|architecture\s+doc(?:ument)?|technical\s+"
        r"(?:design|spec(?:ification)?)|system\s+design|hld|lld|"
        r"low-?level\s+design|high-?level\s+design|rfc)\b", re.I)),
    (DocGoal.PROPOSAL, re.compile(
        r"\b(proposal|project\s+pitch|business\s+case|rfp\s+response|"
        r"statement\s+of\s+work|sow)\b", re.I)),
    (DocGoal.RESEARCH, re.compile(
        r"\b(research\s+(paper|report)|white\s*paper|literature\s+review|"
        r"study\s+report)\b", re.I)),
    (DocGoal.HOW_TO, re.compile(
        r"\b(how-?to|tutorial|walk-?through|step-?by-?step|runbook|"
        r"implementation\s+guide|setup\s+guide|user\s+guide|getting\s+started)\b",
        re.I)),
    (DocGoal.EXECUTIVE_REPORT, re.compile(
        r"\b(executive\s+(summary|report|brief)|for\s+(?:my|the)\s+"
        r"(?:manager|boss|ceo|cto|cfo|vp|director|leadership|board)|"
        r"send\s+(?:this\s+)?to\s+(?:my|the)\s+(?:manager|boss|ceo|cto|"
        r"leadership|board)|status\s+report)\b", re.I)),
]


_GOAL_EXEMPLARS: dict[str, list[str]] = {
    DocGoal.INTERVIEW_NOTES.value: [
        "prepare interview notes", "write up interview feedback",
        "notes from the candidate interview"],
    DocGoal.MEETING_MINUTES.value: [
        "meeting minutes from today's sync", "notes from the meeting",
        "minutes of the standup"],
    DocGoal.TECHNICAL_DESIGN.value: [
        "write a design document for the service", "a system design document",
        "an architecture document", "the technical spec for this"],
    DocGoal.PROPOSAL.value: [
        "draft a client proposal", "a project proposal", "a business case",
        "a statement of work"],
    DocGoal.RESEARCH.value: [
        "write a research paper", "a white paper on this",
        "a literature review", "a research report"],
    DocGoal.HOW_TO.value: [
        "a step by step guide to deploy this", "a how-to tutorial",
        "an implementation guide", "a setup walkthrough"],
    DocGoal.EXECUTIVE_REPORT.value: [
        "an executive summary for my manager", "a status report for leadership",
        "a brief for the executives"],
    DocGoal.GENERAL.value: [
        "explain how this works", "just document this",
        "a general write-up", "put this in a document"],
}


def detect_document_goal(text: str) -> DocGoal:
    """The document's goal SEMANTICALLY (embedding nearest-class); the regex cues
    remain only as the embedder-down fallback."""
    try:
        from app.semantics.gates import classify
        cls = classify(text, _GOAL_EXEMPLARS,
                       cache_key="doc_goal", threshold=0.45)
        if cls is not None:
            return DocGoal(cls)
    except Exception:  # noqa: BLE001
        pass
    for goal, pat in _GOAL_PATTERNS:  # fallback: deterministic cues
        if pat.search(text or ""):
            return goal
    return DocGoal.GENERAL


# ── depth detection ─────────────────────────────────────────────────────────
_QUICK_RE = re.compile(
    r"\b(quick|brief|briefly|short|concise|tl;?dr|high-?level|"
    r"at\s+a\s+glance|one[\s-]?pager|1[\s-]?page|summary|overview)\b", re.I)
_DETAILED_RE = re.compile(
    r"\b(detailed|comprehensive|in-?depth|thorough|exhaustive|full|complete|"
    r"deep-?dive|elaborate|extensive|\d{2,}\s*(?:-?\s*page|pages))\b", re.I)


def detect_depth(text: str) -> Depth:
    t = text or ""
    # A detailed cue wins over a quick one ("a detailed overview" → detailed).
    if _DETAILED_RE.search(t):
        return Depth.DETAILED
    if _QUICK_RE.search(t):
        return Depth.QUICK
    return Depth.MEDIUM


# ── section templates per goal ──────────────────────────────────────────────
def _S(title: str, required: bool = True, desc: str = "") -> PlannedSection:
    return PlannedSection(title=title, required=required, description=desc)

_TEMPLATES: dict[DocGoal, list[PlannedSection]] = {
    DocGoal.TECHNICAL_DESIGN: [
        _S("Overview"), _S("Goals & Non-Goals", False),
        _S("Architecture"), _S("Components"),
        _S("Data Model", False), _S("Implementation"),
        _S("API / Interfaces", False), _S("Deployment", False),
        _S("Security", False), _S("Scalability & Performance", False),
        _S("Alternatives Considered", False), _S("References", False),
    ],
    DocGoal.EXECUTIVE_REPORT: [
        _S("Executive Summary"), _S("Background", False),
        _S("Key Findings"), _S("Impact", False),
        _S("Recommendations"), _S("Next Steps", False),
    ],
    DocGoal.INTERVIEW_NOTES: [
        _S("Summary"), _S("Candidate / Role", False),
        _S("Questions & Answers"), _S("Strengths"),
        _S("Concerns", False), _S("Verdict"),
    ],
    DocGoal.PROPOSAL: [
        _S("Executive Summary"), _S("Problem Statement"),
        _S("Proposed Solution"), _S("Scope", False),
        _S("Timeline", False), _S("Cost / Resources", False),
        _S("Risks & Mitigations", False), _S("Conclusion"),
    ],
    DocGoal.RESEARCH: [
        _S("Abstract"), _S("Introduction"), _S("Background", False),
        _S("Methodology"), _S("Findings"), _S("Discussion", False),
        _S("Conclusion"), _S("References"),
    ],
    DocGoal.HOW_TO: [
        _S("Overview"), _S("Prerequisites", False), _S("Steps"),
        _S("Examples", False), _S("Troubleshooting", False),
        _S("References", False),
    ],
    DocGoal.MEETING_MINUTES: [
        _S("Attendees", False), _S("Agenda", False), _S("Discussion"),
        _S("Decisions"), _S("Action Items"),
    ],
    DocGoal.GENERAL: [],   # no imposed structure — the answer's own stands
}

# Depth → (include optional sections?, pages-per-section heuristic).
_DEPTH_PLAN = {
    Depth.QUICK: (False, 1),
    Depth.MEDIUM: (True, 2),
    Depth.DETAILED: (True, 5),
}


def plan_blueprint(goal: DocGoal, depth: Depth = Depth.MEDIUM) -> Blueprint:
    """The section set for a goal, scaled by depth. QUICK drops optional
    sections; DETAILED keeps everything and estimates more pages each."""
    include_optional, pages_each = _DEPTH_PLAN[depth]
    template = _TEMPLATES.get(goal, [])
    sections = [s for s in template if s.required or include_optional]
    est = max(1, len(sections) * pages_each) if sections else 0
    return Blueprint(goal=goal, depth=depth, sections=sections, est_pages=est)


def plan_document(text: str) -> Blueprint:
    """Detect the goal + depth from the request and build its blueprint."""
    return plan_blueprint(detect_document_goal(text), detect_depth(text))


__all__ = [
    "DocGoal", "Depth", "PlannedSection", "Blueprint",
    "detect_document_goal", "detect_depth", "plan_blueprint", "plan_document",
]
