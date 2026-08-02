"""
Hybrid question classifier.

Fast path: regex/heuristic. Slow path: a tiny LLM (Haiku-class) call that
confirms and tags the type. The two paths combine into one async `classify`
function that returns a `QuestionMeta`. This is consumed by the orchestrator
to pick the tool set for the answer.

Why two paths: heuristics give instant signal so the UI can mark "question
detected" within ~1ms of the utterance arriving. The LLM call refines the
classification but the user doesn't have to wait on it for the chip to light up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.core import lexicons
from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.core.prompt import fill

QuestionType = Literal[
    "behavioral",
    "technical_concept",
    "coding",
    "clarification",
    "smalltalk",
    "unknown",
]

@dataclass
class QuestionMeta:
    """Structured output of the question detection pipeline."""
    is_question: bool
    type: QuestionType
    is_followup: bool
    topic: str
    confidence: float  # 0.0–1.0; combines heuristic + LLM agreement
    source: str        # "heuristic" | "llm" | "hybrid"

# ---- Fast heuristic path -----------------------------------------------
# Hint DATA lives in the central registry (app/core/lexicons.py).
_INTERROGATIVES = lexicons.QD_INTERROGATIVES
_CODING_HINTS = lexicons.QD_CODING_HINTS
_BEHAVIORAL_HINTS = lexicons.QD_BEHAVIORAL_HINTS
_SMALLTALK_HINTS = lexicons.QD_SMALLTALK_HINTS
_FOLLOWUP_STARTERS = lexicons.QD_FOLLOWUP_STARTERS
_AUXILIARIES = frozenset(lexicons.QD_AUXILIARIES)
_IMPERATIVE_PROMPTS = lexicons.QD_IMPERATIVE_PROMPTS

# Openers that are genuinely two-faced: the SAME words open a request for an
# answer and a piece of the speaker's own housekeeping. "Give me an example of
# Kafka" asks; "give me one moment, my screen froze" does not. Only these get
# the semantic floor-holding veto — applying it to wh-words discarded real
# questions.
_AMBIGUOUS_OPENERS = ("give me", "tell", "walk", "let me", "show me", "run me",
                      "take me", "talk me", "help me understand")

# Clause boundaries: a comma, or a coordinating conjunction that joins two
# independent clauses. Splitting on these is what lets an interrogative be found
# where it actually sits rather than only at position 0.
_CLAUSE_SPLIT = re.compile(r"[,;]|\s+(?:and|but|or|so|then)\s+")
_WORD = re.compile(r"[a-z0-9']+")

# Tokens that may follow an inverted auxiliary and still leave it a question.
# A determiner/pronoun/quantifier after "is"/"does" marks subject-auxiliary
# inversion ("is THE cache…", "does IT scale"). Without this guard a statement
# beginning on an auxiliary would read as a question.
_SUBJECT_LEAD = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "i", "there", "your", "my", "our", "his", "her",
    "their", "its", "any", "all", "some", "every", "each", "most", "one",
    "two", "both", "either", "neither", "no", "another", "such",
})


# A follow-up opens on a conjunction and continues the prior thread ("And what
# about failures", "So how does that scale"). Stripping it exposes the
# interrogative underneath; without this the whole utterance reads as starting
# on "and" and the question is invisible.
_LEADING_CONJ = re.compile(r"^(?:and|but|or|so|then|also|okay|ok|alright|now)\b[\s,]*")


def _clauses(t: str) -> list[str]:
    """Split an utterance into clauses. Always returns at least the whole text,
    so a clause-free utterance behaves exactly as before."""
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(t) if p and p.strip()]
    out = []
    for p in parts or [t]:
        out.append(p)
        stripped = _LEADING_CONJ.sub("", p, count=1).strip()
        if stripped and stripped != p:
            out.append(stripped)
    return out


def _opens_with(clause: str, openers) -> bool:
    return any(clause.startswith(w + " ") or clause.startswith(w + "'")
               or clause == w for w in openers)


def _is_inverted(clause: str) -> bool:
    """Subject-auxiliary inversion: AUX + (subject-ish token). English's other
    way of asking without a wh-word, and invisible once STT drops the '?'."""
    words = _WORD.findall(clause)
    if len(words) < 3 or words[0] not in _AUXILIARIES:
        return False
    nxt = words[1]
    # A pronoun/determiner subject, or a capitalized-in-speech topic noun that is
    # not itself another verb — approximated as "not a second auxiliary", which
    # keeps "is is" style STT noise out.
    return nxt in _SUBJECT_LEAD or nxt not in _AUXILIARIES


def heuristic_classify(text: str) -> QuestionMeta:
    """Pattern-only classification. Always returns a meta with `source='heuristic'`."""
    t = text.strip().lower()
    if len(t) < cfg.question_detection.min_question_length:
        return QuestionMeta(False, "unknown", False, "", 0.1, "heuristic")

    by_mark = t.endswith("?")
    # An interrogative opens a CLAUSE, not only the utterance. "Say you inherit
    # a system using Kafka, where do you start" is a question whose interrogative
    # is the second clause; matching only at position 0 missed every utterance of
    # that shape, and with STT dropping the '?' the interviewer got no answer at
    # all. Clauses are split on commas and coordinating conjunctions — cheap,
    # deterministic, and it generalizes instead of enumerating phrasings.
    clauses = _clauses(t)
    by_prefix = any(_opens_with(c, _INTERROGATIVES) for c in clauses)
    # Subject-auxiliary inversion — the other way English forms a question
    # without a wh-word ("Is Kafka durable", "Does it scale horizontally").
    # Requires a following subject token so a statement that merely starts on a
    # verb ("Is is a keyword...") does not qualify.
    by_inversion = any(_is_inverted(c) for c in clauses)
    # Imperative prompts that request an answer or an artifact ("Write a function
    # that…", "Talk me through…", "Rate your comfort with…"). These never carry
    # a '?' and are a large share of what an interviewer actually says.
    by_imperative = any(_opens_with(c, _IMPERATIVE_PROMPTS) for c in clauses)
    is_question = by_mark or by_prefix or by_inversion or by_imperative
    # Did the match come ONLY from a two-faced imperative opener? That is the
    # single case where a semantic floor-holding veto is warranted (below).
    _ambiguous = (by_prefix or by_imperative) and any(
        _opens_with(c, _AMBIGUOUS_OPENERS) for c in clauses)
    # `_INTERROGATIVES` holds imperative openers too ("give me", "tell me"), so
    # the speaker handling their OWN business trips them: "give me one moment,
    # my screen froze" is not a question. Veto semantically — a literal list
    # cannot separate "give me one moment" from "give me an example". Only
    # consulted for a PREFIX match with no '?' (a minority of utterances), so
    # the hot path is unaffected; fail-open ⇒ today's behaviour.
    # Scoped to the AMBIGUOUS openers only. "give me"/"tell"/"walk" are genuinely
    # two-faced — "give me an example" asks, "give me one moment" does not — so a
    # semantic veto earns its place there. A wh-word or an inversion is not
    # ambiguous, and vetoing those cost real questions ("How do you debug an
    # issue with write-ahead logging" was being discarded as floor-holding).
    if _ambiguous and not by_mark and not by_inversion:
        try:
            from app.live.implicit import holds_floor
            if holds_floor(t):
                is_question = False
        except Exception:  # noqa: BLE001 — heuristic must never raise
            pass
    # Indirect and hypothetical probes read as statements syntactically but
    # demand an answer: "I'd like to hear about your project", "Suppose one
    # service goes down." The live decision engine promotes these too; the
    # heuristic must agree so the LLM-down fallback doesn't drop them.
    if not is_question:
        try:
            from app.live.implicit import detect_hypothetical, detect_implicit
            if (detect_implicit(t).is_implicit_question
                    or detect_hypothetical(t).is_implicit_question):
                is_question = True
        except Exception:  # noqa: BLE001 — heuristic must never raise
            pass

    qtype: QuestionType = "unknown"
    if any(h in t for h in _CODING_HINTS):
        qtype = "coding"
    elif any(h in t for h in _BEHAVIORAL_HINTS):
        qtype = "behavioral"
    elif any(h in t for h in _SMALLTALK_HINTS):
        qtype = "smalltalk"
        is_question = False
    elif re.search(r"\b(what is|how does|why is|explain|difference between)\b", t):
        qtype = "technical_concept"

    # Follow-up: a question that OPENS on a conjunction / back-reference
    # ("And why is that?", "So how does that scale?") continues the prior
    # thread. Deterministic signal so the heuristic path (LLM down / fast
    # path) tags follow-ups instead of always reporting False.
    is_followup = is_question and any(
        t.startswith(s) for s in _FOLLOWUP_STARTERS)

    # Confidence must express QUESTION-ness, not whether we could also name the
    # topic type. Conflating them sent unmistakable questions down the SLOW
    # LLM-detection path ("How do partitions work in Kafka?" scored 0.4 — below
    # the fast-path bar — only because "how do" isn't in the narrow qtype regex,
    # while "What is X?" scored 0.7). A terminal '?' is the strongest question
    # signal there is; an unnamed topic says nothing about whether it's a
    # question. Measured by app/eval/live_bench `fast_path_coverage`.
    #
    # A '?' is not the ONLY unambiguous signal. A clause-leading interrogative
    # ("…, where do you start"), subject-auxiliary inversion ("Does it scale")
    # and an imperative prompt ("Write a function that…") are grammatical facts,
    # not guesses — and since STT drops the '?' constantly, they carry most of
    # the recall. Scoring them 0.4 sent a correctly-detected question down the
    # SLOW LLM-confirmation path and added a whole round-trip of latency to the
    # answer, for nothing. Measured: fast_path_coverage on the 4213-row corpus.
    #
    # The SEMANTIC-only promotions (implicit/hypothetical gates) deliberately
    # stay at 0.4. They are similarity judgements rather than grammar, they are
    # where every remaining false positive comes from, and LLM confirmation is
    # exactly what should adjudicate them.
    _structural = by_mark or by_prefix or by_inversion or by_imperative
    if is_question and (_structural or qtype != "unknown"):
        confidence = 0.7
    else:
        confidence = 0.4
    return QuestionMeta(is_question, qtype, is_followup, "", confidence,
                        "heuristic")

# ---- LLM-confirmed path ------------------------------------------------
_CLASSIFIER_PROMPT = """Classify this utterance from a job interview.

Recent interviewer questions (most recent last):
{recent_questions}

Utterance: "{utterance}"

Return ONLY a JSON object with these keys:
- is_question (bool): true if this is an interview question directed at the candidate
- type (string): one of "behavioral", "technical_concept", "coding", "clarification", "smalltalk"
- is_followup (bool): true if this looks like a follow-up to the most recent question
- topic (string): a 1-3 word topic tag (e.g. "kafka", "leadership", "binary tree")
"""

async def classify(
    utterance: str,
    recent_qs: list[str],
    *,
    audio_np=None,
    sample_rate: int = 16_000,
) -> QuestionMeta:
    """Run heuristic + LLM and merge into a single QuestionMeta.

    When `audio_np` is supplied, the prosody analyzer's score is fused
    with the text+context scores (Architecture.md §"Multi-modal
    question detection"). Without audio, the existing heuristic+LLM
    flow runs unchanged.

    If the LLM is configured off (`use_llm_classifier: false`) or unreachable,
    falls back to the heuristic result.
    """
    fast = heuristic_classify(utterance)
    if not cfg.question_detection.use_llm_classifier:
        return _maybe_fuse_prosody(fast, audio_np, sample_rate, recent_qs)

    recent_block = (
        "\n".join(f"- {q}" for q in recent_qs[-cfg.question_detection.recent_q_window :])
        or "(none)"
    )
    prompt = fill(_CLASSIFIER_PROMPT, 
        recent_questions=recent_block, utterance=utterance.replace('"', "'")
    )
    messages = [{"role": "user", "content": prompt}]
    model = cfg.llm.classifier_model or cfg.llm.model

    try:
        raw = await llm.chat_json(messages, model=model)
    except LLMError:
        # LLM unreachable -> trust the heuristic answer.
        return fast

    parsed = _parse_lenient_json(raw)
    if not isinstance(parsed, dict):
        return fast

    is_q = bool(parsed.get("is_question", fast.is_question))
    qtype = parsed.get("type", fast.type)
    if qtype not in ("behavioral", "technical_concept", "coding", "clarification", "smalltalk"):
        qtype = fast.type
    is_fu = bool(parsed.get("is_followup", False))
    topic = str(parsed.get("topic", "")).strip()[:60]

    # Confidence: 0.95 when heuristic and LLM agree on is_question, 0.7 otherwise.
    confidence = 0.95 if is_q == fast.is_question else 0.7

    merged = QuestionMeta(
        is_question=is_q,
        type=qtype,  # type: ignore[arg-type]
        is_followup=is_fu,
        topic=topic,
        confidence=confidence,
        source="hybrid",
    )
    return _maybe_fuse_prosody(merged, audio_np, sample_rate, recent_qs)

def _maybe_fuse_prosody(
    meta: "QuestionMeta", audio_np, sample_rate: int, recent_qs: list[str]
) -> "QuestionMeta":
    """If audio is supplied, blend the prosody score with the text +
    context scores per Architecture.md's 0.55/0.30/0.15 recipe.

    Never raises — prosody is opportunistic. When the analyzer fails
    or no audio is provided, `meta` passes through unchanged.
    """
    if audio_np is None:
        return meta
    try:
        from .prosody_analyzer import analyze
        from .fusion import fuse
    except Exception:  # noqa: BLE001
        return meta
    try:
        feats = analyze(audio_np, sample_rate=sample_rate)
    except Exception:  # noqa: BLE001
        return meta
    # Text score: scale meta.confidence by whether we already think
    # it's a question. Context: bumps when there's a recent question
    # in the window (a follow-up is more likely a question).
    text_score = meta.confidence if meta.is_question else max(0.0, 1.0 - meta.confidence)
    context_score = 0.6 if recent_qs else 0.4
    decision = fuse(
        text_score=text_score,
        prosody_score=feats.is_question_acoustic,
        context_score=context_score,
    )
    return QuestionMeta(
        is_question=decision.is_question,
        type=meta.type,
        is_followup=meta.is_followup,
        topic=meta.topic,
        confidence=decision.score,
        source="multimodal" if meta.source == "hybrid" else f"{meta.source}+prosody",
    )

def _parse_lenient_json(text: str) -> dict | None:
    """Same lenient JSON parsing as profile_builder. Local models drift."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
