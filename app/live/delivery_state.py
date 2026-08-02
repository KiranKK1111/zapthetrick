"""Solo mode: tell a candidate DELIVERING an answer from one ASKING a question.

The problem
-----------
Standard mode separates the two speakers by role: `_role == "candidate"` means
absorb, never answer. Solo mode deliberately turns that off — there is one voice
and nothing to diarize — so its only protection was content echo matching, which
catches the tester *reading our shown answer back*.

That leaves the common case uncovered. A candidate who answers in their **own
words** matches no shown text and carries no role tag, so if the utterance parses
as a question at all it gets transcribed, sent to the LLM, and answered. The app
interrupts the person it is meant to be helping.

The idea
--------
After an answer is displayed the candidate is *expected* to speak. So the default
inverts: within that window, speech is DELIVERY unless it is clearly a new
question. Three independent signals, cheapest first, each usable alone:

1. **Expectation window** — for a few seconds after an answer renders, demand a
   *strong* question signal (a clause-leading interrogative or subject-auxiliary
   inversion — grammar, not similarity) rather than the semantic promotions that
   produce nearly all false positives.
2. **Answer shape** — first-person explanatory speech ("so what I did was…",
   "in my last role…") is structurally different from a question, and cheap to
   recognise deterministically.
3. **Topic continuity** — speech about the topic just answered is delivery; a
   topic shift is a new question.

Only the FIRST layer can suppress on its own. The other two are corroboration,
because the cost of the two errors is wildly asymmetric: wrongly answering while
someone speaks is annoying, but wrongly suppressing a real question leaves them
silent in an interview. Everything here fails open.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# How long after an answer is shown the candidate is expected to be speaking.
# Long enough to cover delivering a real answer, short enough that the next
# genuine question is not caught in it.
DEFAULT_WINDOW_S = 25.0

# First-person openers that mark speech as an ANSWER being delivered. A closed,
# literal set — these are grammatical person markers, not intents.
_ANSWER_OPENERS = (
    "i ", "i'", "we ", "we'", "my ", "our ", "so i", "so we", "well i",
    "yeah so", "yes so", "basically i", "in my", "at my", "when i", "what i",
    "the way i", "for me", "personally", "in our", "on my",
)

# Phrases that mark thinking-aloud rather than either asking or answering.
_THINKING = (
    "let me think", "give me a second", "hmm", "um", "uh", "one moment",
    "let me see", "how do i put",
)

_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class DeliveryState:
    """Per-session record of when an answer was last shown, and about what."""

    shown_at: float = 0.0
    topic: str = ""
    window_s: float = DEFAULT_WINDOW_S
    # Diagnostics — a gate that suppresses invisibly is undebuggable.
    suppressed: int = 0
    admitted: int = 0
    _recent: list[str] = field(default_factory=list, repr=False)

    def answer_shown(self, topic: str = "", now: float | None = None) -> None:
        self.shown_at = now if now is not None else time.monotonic()
        if topic:
            self.topic = topic.strip().lower()

    def in_window(self, now: float | None = None) -> bool:
        if self.shown_at <= 0:
            return False
        t = now if now is not None else time.monotonic()
        return (t - self.shown_at) <= self.window_s


def looks_like_delivery(text: str) -> bool:
    """First-person explanatory speech — someone answering, not asking."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(t.startswith(p) for p in _ANSWER_OPENERS):
        return True
    if any(p in t for p in _THINKING):
        return True
    # A first-person pronoun early in a long utterance, with no interrogative
    # lead, reads as narration.
    words = _WORD.findall(t)
    if len(words) >= 8 and any(w in ("i", "we", "my", "our") for w in words[:6]):
        return True
    return False


# Subject pronouns that turn a wh-word into a CLEFT rather than a question.
_CLEFT_SUBJECTS = frozenset({"i", "we", "you", "he", "she", "they", "it"})


def _is_cleft(clause: str) -> bool:
    """`what I did was …` states; `what did I do` asks.

    English marks the difference by word order: a wh-word followed directly by a
    SUBJECT (wh + subject + verb) is a relative/cleft construction, while a
    question inverts it (wh + auxiliary + subject). Without this, "So what I did
    was shard the topic" reads as an interrogative and the candidate's own
    answer gets sent back to the model as a question.
    """
    words = _WORD.findall(clause or "")
    if len(words) < 3:
        return False
    if words[0] not in ("what", "how", "why", "where", "when", "which", "who"):
        return False
    return words[1] in _CLEFT_SUBJECTS


def has_strong_question_signal(text: str) -> bool:
    """GRAMMAR, not similarity.

    A '?', a clause-leading interrogative, subject-auxiliary inversion or an
    imperative prompt. Deliberately excludes the semantic gates: those are where
    essentially every false positive comes from, and inside the delivery window
    a false positive is exactly what must not happen.
    """
    try:
        from app.question_detection.classifier import (
            _clauses, _is_inverted, _opens_with, _IMPERATIVE_PROMPTS,
            _INTERROGATIVES,
        )
        t = (text or "").strip().lower()
        if not t:
            return False
        if t.endswith("?"):
            return True
        clauses = [c for c in _clauses(t) if not _is_cleft(c)]
        return (any(_opens_with(c, _INTERROGATIVES) for c in clauses)
                or any(_is_inverted(c) for c in clauses)
                or any(_opens_with(c, _IMPERATIVE_PROMPTS) for c in clauses))
    except Exception:  # noqa: BLE001 — fail open: treat as a question
        return True


def topic_continues(text: str, topic: str) -> bool:
    """Whether the utterance is still about the topic just answered.

    Word overlap, not embeddings: this runs on the hot path and only has to
    separate "still on Kafka" from "now let's talk about Redis".
    """
    t = (topic or "").strip().lower()
    if not t:
        return False
    topic_words = {w for w in _WORD.findall(t) if len(w) > 3}
    if not topic_words:
        return False
    said = set(_WORD.findall((text or "").lower()))
    return bool(topic_words & said)


def should_suppress(state: DeliveryState, text: str,
                    now: float | None = None) -> tuple[bool, str]:
    """Should this utterance be treated as DELIVERY (not sent to the LLM)?

    Returns `(suppress, reason)`. The reason is recorded so a suppressed turn is
    explainable rather than a silent disappearance.

    The asymmetry is deliberate throughout: a strong grammatical question signal
    ALWAYS wins, because leaving a real question unanswered in an interview is
    far worse than answering one sentence of a delivery.
    """
    t = (text or "").strip()
    if not t:
        return False, ""
    if not state.in_window(now):
        state.admitted += 1
        return False, "outside the delivery window"
    if has_strong_question_signal(t):
        state.admitted += 1
        return False, "strong question signal"
    if looks_like_delivery(t):
        state.suppressed += 1
        return True, "first-person delivery"
    if topic_continues(t, state.topic):
        state.suppressed += 1
        return True, "continues the answered topic"
    # In the window, no question signal, but nothing positively marks it as
    # delivery either. Admit — an unclear utterance is more safely answered than
    # swallowed.
    state.admitted += 1
    return False, "unclassified — admitted"


# ── Per-session store ───────────────────────────────────────────────────────

_STATES: dict[str, DeliveryState] = {}


def state_for(sid: str) -> DeliveryState:
    if sid not in _STATES:
        _STATES[sid] = DeliveryState()
    return _STATES[sid]


def forget(sid: str) -> None:
    _STATES.pop(sid, None)


def enabled() -> bool:
    """Solo-mode only, and flag-gated. Off ⇒ exactly today's behaviour."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "solo_delivery_state", True))
    except Exception:  # noqa: BLE001
        return True


__all__ = [
    "DeliveryState", "should_suppress", "looks_like_delivery",
    "has_strong_question_signal", "topic_continues", "state_for", "forget",
    "enabled", "DEFAULT_WINDOW_S",
]
