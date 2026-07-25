"""Coach mode — role-flip mock interview (vNext §9.6, Stage 11 Component C).

The copilot flips roles and runs a practice interview: it ASKS (a question arc
built from the candidate's profile/topics), LISTENS, SCORES the answer against the
target band's structural CONTRACT (§4.3), decides a FOLLOW-UP (drill vs advance),
and at the end produces a debrief. `mock.py` already generates practice questions
and `band_contract.py` holds the per-band answer shapes; this module owns the arc
+ scoring + follow-up + session loop.

Deterministic core (the semantic scoring is an INJECTED `judge_fn` seam; a
keyword/length fallback runs without a model). Pure + fail-open. Flag-gated
(`live.coach_mode`, default OFF → no coach surface).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Arc phases (ascending pressure).
WARM_UP = "warm_up"
CORE = "core"
DEEP = "deep"
CURVEBALL = "curveball"
_PHASES = (WARM_UP, CORE, DEEP, CURVEBALL)

# Follow-up actions.
DRILL = "drill"          # weak → probe deeper on the same topic
ADVANCE = "advance"      # solid → next arc question
MOVE_ON = "move_on"      # good enough / time → skip ahead


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "coach_mode", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Question-arc selector
# --------------------------------------------------------------------------- #
@dataclass
class ArcQuestion:
    question: str
    phase: str
    order: int
    category: str = ""

    def to_dict(self) -> dict:
        return {"question": self.question, "phase": self.phase,
                "order": self.order, "category": self.category}


# How many of each phase a full arc aims for (scaled to the pool size).
_ARC_SHAPE = ((WARM_UP, 1), (CORE, 2), (DEEP, 2), (CURVEBALL, 1))


def build_arc(questions, *, band: str = "mid", max_len: int = 6) -> "list[ArcQuestion]":
    """Order a pool of practice questions (each `{question, category, label}` or a
    string) into an ascending-pressure ARC: warm-up → core → deep → curveball.
    Introduction/behavioural questions lead; technical/design ones fill the core
    and deep phases. Never raises → []."""
    try:
        pool = []
        for q in questions or []:
            if isinstance(q, dict):
                pool.append((q.get("question", ""), q.get("category", "")))
            elif isinstance(q, str) and q.strip():
                pool.append((q.strip(), ""))
        pool = [(q, c) for q, c in pool if q]
        if not pool:
            return []
        # Warm-up = introduction/behavioural; the rest by original order.
        intro = [p for p in pool if p[1] in ("introduction", "behavioural")]
        rest = [p for p in pool if p not in intro]
        ordered = intro + rest
        arc: list[ArcQuestion] = []
        phase_plan: list[str] = []
        for phase, n in _ARC_SHAPE:
            phase_plan += [phase] * n
        for i, (q, cat) in enumerate(ordered[:max_len]):
            phase = phase_plan[i] if i < len(phase_plan) else CURVEBALL
            arc.append(ArcQuestion(question=q, phase=phase, order=i, category=cat))
        return arc
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Answer scoring vs the band-contract rubric
# --------------------------------------------------------------------------- #
@dataclass
class AnswerScore:
    score: int                    # 0..100
    coverage: float               # fraction of contract sections addressed
    included: bool                # all `must_include` present
    disciplined: bool             # no `avoid` claims
    concise: bool                 # within the band's time budget (word proxy)
    gaps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "coverage": round(self.coverage, 3),
                "included": self.included, "disciplined": self.disciplined,
                "concise": self.concise, "gaps": list(self.gaps)}


def _mentions(answer: str, phrase: str) -> bool:
    """A lenient 'does the answer touch this' check: ≥60% of the phrase's
    meaningful words appear."""
    words = [w for w in re.findall(r"\w+", (phrase or "").lower()) if len(w) > 2]
    if not words:
        return True
    a = (answer or "").lower()
    return sum(1 for w in words if w in a) / len(words) >= 0.6


def score_answer(answer: str, contract, *, judge_fn=None) -> AnswerScore:
    """Score `answer` against a band `contract` (duck-typed: `sections`,
    `must_include`, `avoid`, `max_seconds`). Deterministic rubric — coverage of
    the expected sections, presence of required elements, absence of over-claims,
    concision within the band — or an INJECTED `judge_fn(answer, contract) ->
    dict` for semantic scoring. Never raises → a zero score."""
    try:
        if judge_fn is not None:
            try:
                obj = judge_fn(answer, contract) or {}
                return AnswerScore(
                    score=int(obj.get("score", 0)),
                    coverage=float(obj.get("coverage", 0.0)),
                    included=bool(obj.get("included", False)),
                    disciplined=bool(obj.get("disciplined", True)),
                    concise=bool(obj.get("concise", True)),
                    gaps=list(obj.get("gaps", [])))
            except Exception:  # noqa: BLE001
                pass
        text = answer or ""
        sections = tuple(getattr(contract, "sections", ()) or ())
        must = tuple(getattr(contract, "must_include", ()) or ())
        avoid = tuple(getattr(contract, "avoid", ()) or ())
        max_seconds = float(getattr(contract, "max_seconds", 45) or 45)

        gaps: list[str] = []
        # Coverage of the expected answer sections.
        covered = [s for s in sections if _mentions(text, s)]
        coverage = (len(covered) / len(sections)) if sections else 1.0
        for s in sections:
            if s not in covered:
                gaps.append(f"missing: {s}")
        # Required elements.
        missing_req = [m for m in must if not _mentions(text, m)]
        included = not missing_req
        gaps += [f"needs: {m}" for m in missing_req]
        # Discipline (over-claims to avoid).
        over = [a for a in avoid if _mentions(text, a)]
        disciplined = not over
        gaps += [f"over-claim: {a}" for a in over]
        # Concision: ~2.5 words/second budget from the band time.
        words = len(re.findall(r"\w+", text))
        concise = words <= max_seconds * 2.5 * 1.2 or words == 0

        base = 100.0 * coverage
        if not included:
            base -= 20
        if not disciplined:
            base -= 25
        if not concise:
            base -= 10
        score = int(max(0, min(100, base)))
        return AnswerScore(score, coverage, included, disciplined, concise, gaps)
    except Exception:  # noqa: BLE001
        return AnswerScore(0, 0.0, False, True, True, ["scoring error"])


# --------------------------------------------------------------------------- #
# Follow-up decision
# --------------------------------------------------------------------------- #
@dataclass
class CoachStep:
    action: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason}


def follow_up_decision(score: AnswerScore, *, drill_below: int = 60,
                       move_on_above: int = 85) -> CoachStep:
    """From an answer score, choose the next move: DRILL a weak answer, MOVE_ON
    from a strong one, else ADVANCE. Discipline failures always drill (an
    over-claim is worth correcting). Never raises."""
    try:
        if not score.disciplined:
            return CoachStep(DRILL, "over-claim — worth correcting")
        if score.score < drill_below:
            return CoachStep(DRILL, f"weak ({score.score}) - probe deeper")
        if score.score >= move_on_above:
            return CoachStep(MOVE_ON, f"strong ({score.score}) - move on")
        return CoachStep(ADVANCE, f"solid ({score.score}) - next question")
    except Exception:  # noqa: BLE001
        return CoachStep(ADVANCE, "error -> advance")


# --------------------------------------------------------------------------- #
# Coach session
# --------------------------------------------------------------------------- #
@dataclass
class CoachSession:
    band: str = "mid"
    arc: list = field(default_factory=list)       # list[ArcQuestion]
    pos: int = 0
    scores: list = field(default_factory=list)    # list[(ArcQuestion, AnswerScore)]

    def current(self) -> "ArcQuestion | None":
        return self.arc[self.pos] if 0 <= self.pos < len(self.arc) else None

    def record(self, score: AnswerScore) -> CoachStep:
        """Record the current answer's score + choose the next step. ADVANCE /
        MOVE_ON moves the cursor; DRILL stays. Never raises."""
        try:
            q = self.current()
            if q is not None:
                self.scores.append((q, score))
            step = follow_up_decision(score)
            if step.action in (ADVANCE, MOVE_ON):
                self.pos += 1
            return step
        except Exception:  # noqa: BLE001
            self.pos += 1
            return CoachStep(ADVANCE, "error")

    def done(self) -> bool:
        return self.pos >= len(self.arc)

    def debrief(self) -> dict:
        """A debrief summary: per-question scores, the average, and the recurring
        gaps to practice. Descriptive (never a pass/fail verdict). Never raises."""
        try:
            if not self.scores:
                return {"band": self.band, "answered": 0, "average": 0,
                        "focus_areas": []}
            avg = round(sum(s.score for _, s in self.scores) / len(self.scores))
            gap_counts: dict = {}
            for _, s in self.scores:
                for g in s.gaps:
                    gap_counts[g] = gap_counts.get(g, 0) + 1
            focus = sorted(gap_counts, key=lambda g: -gap_counts[g])[:5]
            return {"band": self.band, "answered": len(self.scores),
                    "average": avg, "focus_areas": focus,
                    "per_question": [{"question": q.question, "score": s.score}
                                     for q, s in self.scores]}
        except Exception:  # noqa: BLE001
            return {"band": self.band, "answered": 0, "average": 0, "focus_areas": []}


__all__ = ["WARM_UP", "CORE", "DEEP", "CURVEBALL", "DRILL", "ADVANCE", "MOVE_ON",
           "enabled", "ArcQuestion", "build_arc", "AnswerScore", "score_answer",
           "CoachStep", "follow_up_decision", "CoachSession"]
