"""
Structured conversational-event typer (live-conversational-intelligence R1, R2).

Wraps the existing single `question_detection.agent.predict` call and widens its
result into a typed `UtteranceEvent`:

  - `kind`      — QUESTION / FOLLOWUP / EXPLANATION / GREETING / SMALL_TALK /
                  TRANSITION / ACKNOWLEDGEMENT / TOPIC_CHANGE / ANSWER_HINT
  - `questions` — one or more questions split from a multi-question utterance
                  ("What is Kafka, why use it, and how do you scale it?")
  - `context`   — leading non-question sentences that precede the question
                  ("We use Kafka. Ordering matters. How do you dedupe?")

Crucially this adds **no second blocking LLM call** — it reuses the one
`agent.predict` result and derives the event/boundary/multi-question split
**deterministically** on top of it. Disabled or on any error it falls back to a
single QUESTION event carrying the transcript (today's behavior).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core import lexicons
from app.question_detection import agent as _agent

# Event kinds.
QUESTION = "QUESTION"
FOLLOWUP = "FOLLOWUP"
EXPLANATION = "EXPLANATION"
GREETING = "GREETING"
SMALL_TALK = "SMALL_TALK"
TRANSITION = "TRANSITION"
ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
TOPIC_CHANGE = "TOPIC_CHANGE"
ANSWER_HINT = "ANSWER_HINT"

ANSWERABLE = {QUESTION, FOLLOWUP}

_INTERROGATIVE = lexicons.LIVE_EVENTS_INTERROGATIVE

_GREETING_CUES = lexicons.LIVE_EVENTS_GREETING_CUES
_ACK_CUES = lexicons.LIVE_EVENTS_ACK_CUES
_TRANSITION_CUES = lexicons.LIVE_EVENTS_TRANSITION_CUES


@dataclass
class UtteranceEvent:
    """A typed unit derived from one finalized utterance."""
    kind: str
    questions: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    topic: str = ""
    difficulty: str = "standard"
    confidence: float = 0.0
    answer_hint: str = ""
    source: str = "agent"
    # Classifier question type (coding | technical_concept | behavioral |
    # clarification | smalltalk) — the accurate intent, surfaced to the UI.
    qtype: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.kind in ANSWERABLE and bool(self.questions)


def _sentences(text: str) -> list[str]:
    """Split text into sentence-ish parts, keeping their terminators."""
    parts = re.split(r"(?<=[.?!])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def _looks_like_question(s: str) -> bool:
    t = s.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first = t.split()[0].rstrip(",.") if t.split() else ""
    return first in _INTERROGATIVE or any(
        t.startswith(w + " ") or t.startswith(w + "'") for w in _INTERROGATIVE
    )


def split_questions(text: str) -> list[str]:
    """Deterministically split a (cleaned) question string into one or more
    questions. Handles both explicit '?' boundaries and conjoined questions
    ("What is Kafka, why use it, and how do you scale it?"). Returns at least
    one item for a non-empty input."""
    t = (text or "").strip()
    if not t:
        return []

    # 1. Multiple explicit '?' terminators → one question each.
    by_q = [p.strip() for p in re.split(r"\?+", t) if p and p.strip()]
    if len(by_q) >= 2:
        return [(p if p.endswith("?") else p + "?") for p in by_q]

    # 2. Single sentence: try to split conjoined questions on commas / "and".
    core = t[:-1] if t.endswith("?") else t
    fragments = re.split(r",\s*and\s+|,\s*|\s+and\s+", core)
    fragments = [f.strip() for f in fragments if f and f.strip()]
    interrog = [f for f in fragments if _looks_like_question(f)]
    if len(interrog) >= 2:
        # Re-question-mark each conjoined interrogative fragment.
        return [(f if f.endswith("?") else f + "?") for f in interrog]

    # 3. Single question.
    return [t]


def split_boundary(raw_utterance: str, question: str) -> tuple[list[str], str]:
    """Separate leading context sentences from the question inside a
    multi-sentence utterance. Returns (context_sentences, question_text).

    "We use Kafka. Ordering matters. How do you dedupe?" ->
        (["We use Kafka.", "Ordering matters."], "How do you dedupe?")
    Best-effort + deterministic; on no clear boundary returns ([], question)."""
    sents = _sentences(raw_utterance)
    if len(sents) <= 1:
        return [], (question or raw_utterance).strip()
    # Trailing question sentence(s) are the question; leading statements context.
    ctx: list[str] = []
    q_sents: list[str] = []
    for s in sents:
        if _looks_like_question(s):
            q_sents.append(s)
        elif q_sents:
            # A statement AFTER a question — keep it with the question block.
            q_sents.append(s)
        else:
            ctx.append(s)
    q_text = " ".join(q_sents).strip() or (question or "").strip()
    return ctx, q_text


def _kind_for(pred: "_agent.Prediction") -> str:
    """Map an agent Prediction (+ light cues) onto an event kind."""
    if pred.is_question:
        return QUESTION
    t = (pred.type or "").lower()
    text = (pred.question or "").lower()  # empty for non-questions; fall through
    if t == "smalltalk":
        # Refine small-talk into greeting / acknowledgement / transition.
        return SMALL_TALK
    return EXPLANATION


def _refine_nonquestion(raw: str, base_kind: str) -> str:
    """Refine a non-question kind from surface cues on the raw transcript."""
    t = (raw or "").strip().lower()
    if not t:
        return base_kind
    if any(c in t for c in _GREETING_CUES):
        return GREETING
    if any(t.startswith(c) or (" " + c) in t for c in _TRANSITION_CUES):
        return TRANSITION
    if any(t.startswith(c) for c in _ACK_CUES) and len(t.split()) <= 6:
        return ACKNOWLEDGEMENT
    return base_kind


async def type_utterance(
    utterance: str,
    recent: list[str],
    audio_np=None,
    *,
    predictor=None,
    domain: str = "",
) -> UtteranceEvent:
    """Type one finalized utterance into a structured `UtteranceEvent`, reusing
    the single `agent.predict` call (no second blocking LLM call). `predictor`
    is injectable for testing; defaults to `question_detection.agent.predict`.
    `domain` is the optional STT-repair context forwarded to the predictor.

    Never raises — on any failure it returns a single QUESTION event carrying
    the transcript (today's fail-open behavior)."""
    text = (utterance or "").strip()
    if not text:
        return UtteranceEvent(kind=SMALL_TALK, questions=[], context=[], source="empty")

    predict = predictor or _agent.predict
    try:
        try:
            pred = await predict(text, recent or [], domain=domain)
        except TypeError:
            # An injected test predictor may not accept `domain` — fall back.
            pred = await predict(text, recent or [])
    except Exception:  # noqa: BLE001 — never drop a turn on predictor error
        return UtteranceEvent(
            kind=QUESTION, questions=[text], context=[], topic="",
            difficulty="standard", confidence=0.5, source="fallback",
        )

    try:
        if pred.is_question:
            cleaned = (pred.question or text).strip()
            context, q_text = split_boundary(text, cleaned)
            questions = split_questions(q_text)
            return UtteranceEvent(
                kind=QUESTION,
                questions=questions,
                context=context,
                topic=pred.topic or "",
                difficulty=getattr(pred, "difficulty", "standard") or "standard",
                confidence=0.9,
                source="agent",
                qtype=getattr(pred, "type", "") or "",
            )
        kind = _refine_nonquestion(text, _kind_for(pred))
        return UtteranceEvent(
            kind=kind, questions=[], context=_sentences(text),
            topic=pred.topic or "", confidence=0.6, source="agent",
        )
    except Exception:  # noqa: BLE001 — derivation must never break the turn
        return UtteranceEvent(
            kind=(QUESTION if pred.is_question else EXPLANATION),
            questions=([pred.question or text] if pred.is_question else []),
            context=[], topic=getattr(pred, "topic", "") or "",
            difficulty=getattr(pred, "difficulty", "standard") or "standard",
            confidence=0.5, source="fallback",
        )
