"""
Interviewer hidden-goal / probe intent (roadmap Phase 2 #9 / 2B-9).

An interviewer's question usually carries a hidden objective beyond its literal
text: are they probing DEPTH ("but why?"), checking FUNDAMENTALS ("what is a
mutex?"), STRESS-testing ("are you sure?"), gauging CULTURE fit ("tell me about
a conflict"), or sizing BREADTH? Surfacing that objective lets the answer meet
the real intent (e.g. a depth-probe wants first-principles, not a definition).

Deterministic keyword+qtype heuristic — no LLM call, computed inline with the
other advisory signals. Purely advisory: surfaced as `meta.interviewer_intent`
plus at most a one-line answer directive; never gates the answer. Fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

DEPTH_PROBE = "depth_probe"
FUNDAMENTALS = "fundamentals_check"
STRESS_TEST = "stress_test"
CULTURE_FIT = "culture_fit"
BREADTH = "breadth_check"
NEUTRAL = "neutral"

# Lowercased cue phrases. Order matters: earlier buckets win on overlap.
_CUES = (
    (STRESS_TEST, ("are you sure", "is that right", "convince me", "prove it",
                   "why do you think that", "defend", "push back", "really?",
                   "that doesn't", "i disagree", "what if you're wrong")),
    (DEPTH_PROBE, ("but why", "go deeper", "dig into", "in detail", "under the hood",
                   "internally", "how exactly", "walk me through how", "trade-off",
                   "tradeoff", "why does", "what happens when", "edge case")),
    (CULTURE_FIT, ("tell me about a time", "conflict", "disagreed", "team",
                   "why do you want", "why us", "why this role", "your weakness",
                   "feedback", "failure", "mistake")),
)

_HIDDEN_GOALS = {
    DEPTH_PROBE: "Assess genuine depth of understanding, not memorized facts.",
    FUNDAMENTALS: "Confirm the candidate has solid fundamentals.",
    STRESS_TEST: "See how the candidate reasons under challenge/pressure.",
    CULTURE_FIT: "Gauge values, collaboration and self-awareness.",
    BREADTH: "Map the breadth of the candidate's experience.",
    NEUTRAL: "",
}

_DIRECTIVES = {
    DEPTH_PROBE: ("This is a DEPTH probe — answer from first principles with a "
                  "concrete mechanism or trade-off, not a surface definition."),
    STRESS_TEST: ("This is a STRESS test — stay composed, acknowledge the point, "
                  "and defend your reasoning with evidence rather than backing "
                  "down reflexively."),
    CULTURE_FIT: ("This is a CULTURE-fit probe — answer with a specific, honest "
                  "example (situation → action → result) and show self-awareness."),
    FUNDAMENTALS: ("This checks FUNDAMENTALS — give a crisp, correct definition "
                   "first, then one illustrating detail."),
    BREADTH: "",
    NEUTRAL: "",
}


@dataclass
class ProbeIntent:
    label: str = NEUTRAL
    hidden_goal: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"probe": self.label, "hidden_goal": self.hidden_goal,
                "confidence": round(self.confidence, 3)}


def probe_intent(question: str, qtype: str | None = None) -> ProbeIntent:
    """Classify the interviewer's hidden objective. Never raises → NEUTRAL."""
    try:
        low = " " + (question or "").lower().strip() + " "
        for label, cues in _CUES:
            if any(c in low for c in cues):
                return ProbeIntent(label=label, hidden_goal=_HIDDEN_GOALS[label],
                                   confidence=0.75)
        qt = (qtype or "").strip().lower()
        if qt == "behavioral":
            return ProbeIntent(label=CULTURE_FIT, hidden_goal=_HIDDEN_GOALS[CULTURE_FIT],
                               confidence=0.6)
        if qt in ("technical_concept", "definition"):
            # A short, single-clause definition question checks fundamentals.
            if len((question or "").split()) <= 8:
                return ProbeIntent(label=FUNDAMENTALS,
                                   hidden_goal=_HIDDEN_GOALS[FUNDAMENTALS],
                                   confidence=0.55)
        return ProbeIntent(label=NEUTRAL, confidence=0.0)
    except Exception:  # noqa: BLE001
        return ProbeIntent(label=NEUTRAL, confidence=0.0)


def directive(intent: ProbeIntent) -> str:
    """A one-line answer directive for the detected probe. '' when neutral/low."""
    try:
        if intent.confidence < 0.5:
            return ""
        return _DIRECTIVES.get(intent.label, "")
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["ProbeIntent", "probe_intent", "directive",
           "DEPTH_PROBE", "FUNDAMENTALS", "STRESS_TEST", "CULTURE_FIT",
           "BREADTH", "NEUTRAL"]
