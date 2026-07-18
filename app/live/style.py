"""
Interviewer-style learning (live-conversational-intelligence R14).

Builds a rolling, per-session estimate of the interviewer's style (question
length, follow-up depth, topic-switch frequency) and offers a small detection-
threshold adjustment so the live module adapts to a rapid-fire vs deep-diver
interviewer. In-process per session (attached to the context tracker); bounded.
Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RAPID_FIRE = "rapid_fire"
DEEP_DIVER = "deep_diver"
BALANCED = "balanced"


@dataclass
class InterviewerStyle:
    questions: int = 0
    followups: int = 0
    topic_switches: int = 0
    total_words: int = 0
    _recent_lens: list = field(default_factory=list)

    def observe(self, *, question: str = "", is_followup: bool = False,
                topic_switch: bool = False) -> None:
        """Record one observed question. Never raises."""
        try:
            q = (question or "").strip()
            if q:
                self.questions += 1
                n = len(q.split())
                self.total_words += n
                self._recent_lens.append(n)
                if len(self._recent_lens) > 20:        # bounded
                    self._recent_lens.pop(0)
            if is_followup:
                self.followups += 1
            if topic_switch:
                self.topic_switches += 1
        except Exception:  # noqa: BLE001
            pass

    def avg_question_words(self) -> float:
        return (self.total_words / self.questions) if self.questions else 0.0

    def followup_rate(self) -> float:
        return (self.followups / self.questions) if self.questions else 0.0

    def switch_rate(self) -> float:
        return (self.topic_switches / self.questions) if self.questions else 0.0

    def label(self) -> str:
        if self.questions < 3:
            return BALANCED
        avg = self.avg_question_words()
        if avg <= 8 and self.followup_rate() >= 0.5:
            return RAPID_FIRE
        if avg >= 18 or self.followup_rate() >= 0.6:
            return DEEP_DIVER
        return BALANCED

    def threshold_adjustment(self) -> float:
        """Small additive nudge to the question-detection threshold: a rapid-
        fire interviewer warrants a lower bar (catch terse questions); a
        deep-diver a slightly higher bar. Bounded to +/-0.1."""
        lbl = self.label()
        if lbl == RAPID_FIRE:
            return -0.08
        if lbl == DEEP_DIVER:
            return 0.05
        return 0.0

    def snapshot(self) -> dict:
        return {
            "style": self.label(),
            "avg_question_words": round(self.avg_question_words(), 1),
            "followup_rate": round(self.followup_rate(), 2),
            "switch_rate": round(self.switch_rate(), 2),
        }


def for_tracker(tracker) -> InterviewerStyle:
    """Return the InterviewerStyle attached to a per-session tracker (lazily)."""
    s = getattr(tracker, "_live_style", None)
    if s is None:
        s = InterviewerStyle()
        try:
            setattr(tracker, "_live_style", s)
        except Exception:  # noqa: BLE001
            pass
    return s


# ── Cognitive-load / pace estimation (R58) ─────────────────────────────
# Estimate the candidate's cognitive load / pace from cheap signals (recent
# question burst rate + answer length they must produce) so the live module can
# adapt answer DEPTH: under high load / rapid pace, prefer shorter, glanceable
# answers; under low load, allow fuller depth.

LOAD_HIGH = "high"
LOAD_MODERATE = "moderate"
LOAD_LOW = "low"


def cognitive_load(*, questions_per_min: float | None = None,
                   pending_answers: int = 0,
                   interviewer_style: str = BALANCED) -> str:
    """Estimate cognitive load. Never raises → MODERATE."""
    try:
        score = 0
        if questions_per_min is not None and questions_per_min >= 6:
            score += 2
        elif questions_per_min is not None and questions_per_min >= 3:
            score += 1
        if pending_answers >= 2:
            score += 1
        if interviewer_style == RAPID_FIRE:
            score += 1
        if score >= 3:
            return LOAD_HIGH
        if score >= 1:
            return LOAD_MODERATE
        return LOAD_LOW
    except Exception:  # noqa: BLE001
        return LOAD_MODERATE


def depth_for_load(load: str) -> str:
    """Map cognitive load to an answer-depth preference directive ('' for low,
    where full depth is fine). Never raises."""
    try:
        if load == LOAD_HIGH:
            return "Candidate is under high cognitive load — keep the answer short and glanceable."
        if load == LOAD_MODERATE:
            return "Keep the answer concise; lead with the key point."
        return ""
    except Exception:  # noqa: BLE001
        return ""
