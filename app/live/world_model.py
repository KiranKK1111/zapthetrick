"""
Interview world model (live-conversational-intelligence R34).

A unified, per-session structured state — current topic/sub-topic, the active
question (+ its `qid`), whether the candidate has answered, whether the
interviewer seems satisfied, and the running assumptions + constraints (e.g. a
system-design round's "assume 100M users", "no third-party services"). Later
answers honor the recorded constraints until superseded. In-process on the
existing per-session tracker (no DB); every collection is bounded. Deterministic
+ fail-open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core import lexicons

_MAX = 24


@dataclass
class InterviewWorldModel:
    topic: str = ""
    subtopic: str = ""
    active_question: str = ""
    active_qid: str = ""
    candidate_answered: bool = False
    interviewer_satisfied: bool = False
    assumptions: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    # Commitments store (dual-source): what each side actually SAID on a topic —
    # e.g. the candidate's stated salary figure, a competing offer, notice
    # period, or an interviewer signal ("that's high"). Keyed by a stable slot
    # name → {role, value, topic}. Bounded. Lets later answers (esp. negotiation)
    # react to what was already said instead of giving generic advice.
    commitments: dict = field(default_factory=dict)

    def set_active(self, question: str, qid: str = "", topic: str = "") -> None:
        self.active_question = (question or "").strip()
        self.active_qid = qid or self.active_qid
        if topic:
            if self.topic and topic.lower() not in self.topic.lower():
                self.subtopic = topic
            else:
                self.topic = topic
        self.candidate_answered = False
        self.interviewer_satisfied = False

    def add_assumption(self, text: str) -> None:
        t = (text or "").strip()
        if t and t not in self.assumptions:
            self.assumptions.append(t)
            if len(self.assumptions) > _MAX:
                self.assumptions.pop(0)

    def add_constraint(self, text: str) -> None:
        t = (text or "").strip()
        if t and t not in self.constraints:
            self.constraints.append(t)
            if len(self.constraints) > _MAX:
                self.constraints.pop(0)

    def mark_candidate_answered(self) -> None:
        self.candidate_answered = True

    def mark_satisfied(self, satisfied: bool) -> None:
        self.interviewer_satisfied = bool(satisfied)

    def honored_context(self) -> str:
        """A directive fragment of the active assumptions + constraints later
        answers must honor ("" when none)."""
        parts = []
        if self.assumptions:
            parts.append("Assumptions: " + "; ".join(self.assumptions[-6:]))
        if self.constraints:
            parts.append("Constraints: " + "; ".join(self.constraints[-6:]))
        return " ".join(parts)

    def snapshot(self) -> dict:
        return {
            "topic": self.topic, "subtopic": self.subtopic,
            "active_question": self.active_question[:120],
            "candidate_answered": self.candidate_answered,
            "interviewer_satisfied": self.interviewer_satisfied,
            "assumptions": list(self.assumptions[-6:]),
            "constraints": list(self.constraints[-6:]),
        }


# ── assumption / constraint extraction (deterministic) ─────────────────
_ASSUMPTION_CUES = lexicons.LIVE_WORLD_ASSUMPTION_CUES
_CONSTRAINT_CUES = lexicons.LIVE_WORLD_CONSTRAINT_CUES


def extract_world(text: str, model: "InterviewWorldModel") -> None:
    """Pull assumptions / constraints from an utterance into the world model.
    Best-effort; never raises."""
    try:
        low = (text or "").lower()
        if any(c in low for c in _ASSUMPTION_CUES):
            model.add_assumption(text.strip())
        if any(c in low for c in _CONSTRAINT_CUES):
            model.add_constraint(text.strip())
    except Exception:  # noqa: BLE001
        pass


def for_tracker(tracker) -> InterviewWorldModel:
    m = getattr(tracker, "_live_world", None)
    if m is None:
        m = InterviewWorldModel()
        try:
            setattr(tracker, "_live_world", m)
        except Exception:  # noqa: BLE001
            pass
    return m


# ── Coreference resolution (R51) ───────────────────────────────────────
# Resolve pronouns / "it" / "that" / "the same" in a follow-up against the
# active question/topic. LOW-confidence resolutions are DEFERRED to the
# clarifier (returned with resolved=False) rather than guessing.
_PRONOUNS = ("it", "that", "this", "they", "them", "those", "the same", "the latter",
             "the former", "the above")


def resolve_coreference(utterance: str, model: "InterviewWorldModel") -> dict:
    """Resolve a coreference against the world model. Returns
    {resolved, referent, confidence, defer}. Never raises."""
    try:
        t = (utterance or "").strip().lower()
        if not t:
            return {"resolved": False, "referent": "", "confidence": 0.0, "defer": False}
        has_pron = any(
            (" " + p + " ") in (" " + t + " ") or t.startswith(p + " ")
            for p in _PRONOUNS
        )
        if not has_pron:
            return {"resolved": False, "referent": "", "confidence": 0.0, "defer": False}
        referent = model.subtopic or model.topic or model.active_question
        if not referent:
            # A pronoun with nothing to bind to → defer to the clarifier.
            return {"resolved": False, "referent": "", "confidence": 0.2, "defer": True}
        # Confidence higher when there's a single clear topic anchor.
        conf = 0.75 if (model.topic and not model.subtopic) else 0.55
        if conf < 0.6:
            return {"resolved": False, "referent": referent, "confidence": conf, "defer": True}
        return {"resolved": True, "referent": referent, "confidence": conf, "defer": False}
    except Exception:  # noqa: BLE001
        return {"resolved": False, "referent": "", "confidence": 0.0, "defer": False}


# ── Recurring-topic Skill_Gap (R57) ────────────────────────────────────
def record_topic(model: "InterviewWorldModel", topic: str) -> None:
    """Track how often a topic recurs across the session (in-process, bounded)."""
    try:
        t = (topic or "").strip().lower()
        if not t:
            return
        counts = getattr(model, "_topic_counts", None)
        if counts is None:
            counts = {}
            model._topic_counts = counts  # type: ignore[attr-defined]
        counts[t] = counts.get(t, 0) + 1
        if len(counts) > 64:  # bounded
            # drop the smallest
            k = min(counts, key=counts.get)
            counts.pop(k, None)
    except Exception:  # noqa: BLE001
        pass


def skill_gaps(model: "InterviewWorldModel", min_count: int = 2) -> list[str]:
    """Topics the interviewer keeps probing (>= min_count) — likely skill gaps
    to reinforce with extra retrieval. Never raises → []."""
    try:
        counts = getattr(model, "_topic_counts", {}) or {}
        return sorted([t for t, n in counts.items() if n >= min_count],
                      key=lambda t: counts[t], reverse=True)
    except Exception:  # noqa: BLE001
        return []


# ── Commitments: what each side actually said on a topic (dual-source) ──
CANDIDATE = "candidate"
INTERVIEWER = "interviewer"


def record_commitment(model: "InterviewWorldModel", slot: str, role: str,
                      value: str, topic: str = "") -> None:
    """Record a commitment/statement in the world model. Later statements on the
    same slot supersede earlier ones. Bounded. Never raises."""
    try:
        s = (slot or "").strip().lower()
        v = (value or "").strip()
        if not s or not v:
            return
        model.commitments[s] = {"role": (role or "").strip().lower(),
                                 "value": v, "topic": (topic or "").strip().lower()}
        if len(model.commitments) > _MAX:
            # Drop an arbitrary oldest key (dict preserves insertion order).
            first = next(iter(model.commitments))
            model.commitments.pop(first, None)
    except Exception:  # noqa: BLE001
        pass


def commitments_for(model: "InterviewWorldModel", topic: str = "") -> dict:
    """Return the commitments relevant to a topic (or all when topic empty).
    Never raises → {}."""
    try:
        if not topic:
            return dict(model.commitments)
        name = topic.strip().lower()
        return {k: v for k, v in model.commitments.items()
                if not v.get("topic") or v["topic"] in name or name in v["topic"]
                or k in name}
    except Exception:  # noqa: BLE001
        return {}


# Deterministic extraction of common interview commitments from an utterance.
_SALARY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(lpa|lakhs?|lac|k|crore|cr|per\s*annum|/\s*year|usd|\$|inr|rupees)",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"(?:\$|₹|usd|inr)\s*\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_NOTICE_RE = re.compile(
    r"notice period(?:\s*(?:of|is))?\s*(\d+\s*(?:days?|weeks?|months?))", re.IGNORECASE)
_OFFER_CUES = ("another offer", "competing offer", "other offer", "counter offer",
               "counter-offer", "an offer from")
# Interviewer pushback signals on a stated ask.
_HIGH_SIGNAL = ("too high", "bit high", "a little high", "above the band",
                "out of range", "over budget", "beyond our range", "that's high",
                "quite high")
_LOW_SIGNAL = ("too low", "below market", "you could ask for more")


def extract_commitments(text: str, role: str = CANDIDATE) -> dict:
    """Pull salary/offer/notice commitments (candidate) or pushback signals
    (interviewer) from an utterance. Returns a {slot: value} dict. Never raises."""
    out: dict = {}
    try:
        t = (text or "").strip()
        if not t:
            return out
        low = t.lower()
        r = (role or CANDIDATE).strip().lower()
        if r == CANDIDATE:
            m = _SALARY_RE.search(low) or _CURRENCY_RE.search(t)
            if m:
                out["salary"] = m.group(0).strip()
            for cue in _OFFER_CUES:
                if cue in low:
                    out["competing_offer"] = "yes"
                    break
            mn = _NOTICE_RE.search(low)
            if mn:
                out["notice_period"] = mn.group(1).strip()
        else:  # interviewer signals
            if any(c in low for c in _HIGH_SIGNAL):
                out["salary_signal"] = "interviewer_thinks_high"
            elif any(c in low for c in _LOW_SIGNAL):
                out["salary_signal"] = "interviewer_thinks_low"
        return out
    except Exception:  # noqa: BLE001
        return out
