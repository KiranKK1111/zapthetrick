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

def heuristic_classify(text: str) -> QuestionMeta:
    """Pattern-only classification. Always returns a meta with `source='heuristic'`."""
    t = text.strip().lower()
    if len(t) < cfg.question_detection.min_question_length:
        return QuestionMeta(False, "unknown", False, "", 0.1, "heuristic")

    is_question = (
        t.endswith("?")
        or any(t.startswith(w + " ") or t.startswith(w + "'") for w in _INTERROGATIVES)
    )
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

    confidence = 0.7 if is_question and qtype != "unknown" else 0.4
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
