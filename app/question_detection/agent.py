"""
LLM question-prediction agent for the Live module.

Given a raw speech-to-text transcript of what the interviewer just said —
which may contain filler, false starts, or recognition errors — a fast LLM
decides whether it's a question directed at the candidate and extracts the
clean, complete question plus its type. The cleaned question is then handed
to the answer model.

This replaces the old heuristic classifier (hardcoded interrogative/keyword
lists): an LLM understands intent, resolves context, and repairs STT errors
far better than keyword matching — and there's nothing hardcoded to maintain.
Routing goes through the auto-router, so it inherits the same multi-provider
fallback the Chat module uses.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.question_detection.classifier import _parse_lenient_json
from app.core.prompt import fill

@dataclass
class Prediction:
    is_question: bool
    question: str   # cleaned, self-contained question (or "" if not a question)
    type: str       # coding | technical_concept | behavioral | clarification | smalltalk
    topic: str
    difficulty: str = "standard"  # trivial | standard | hard | expert

_VALID_TYPES = {
    "coding", "technical_concept", "behavioral", "clarification", "smalltalk",
}

_VALID_DIFFICULTY = {"trivial", "standard", "hard", "expert"}

_PROMPT = """You are assisting a candidate during a LIVE technical interview. Below is a speech-to-text transcript of what the INTERVIEWER just said, produced by an accurate recogniser (Whisper large-v3).

Recent interviewer questions in this session (context, most recent last):
{recent}

{domain}
Transcript: "{transcript}"

TRUST THE TRANSCRIPT. It is usually correct. Your job is light cleanup, NOT rewriting:
1. is_question: true if the interviewer is asking the candidate something that deserves an answer (technical, coding, conceptual, behavioral, or a clarification/follow-up). false if it is a statement, acknowledgement, filler, or small talk.
   Interview questions are often NOT phrased as questions — ALL of these are is_question=true:
   - INDIRECT / imperative probes: "Walk me through your project.", "I'd like to hear how you handled that.", "Tell us about a time you disagreed with a decision."
   - HYPOTHETICAL / assumption scenarios: "Suppose one of your services goes down.", "Let's say traffic doubles overnight.", "Imagine the database is slow." — the interviewer is asking how the candidate would respond to the scenario.
   - EMBEDDED questions mid-explanation: a question buried inside surrounding commentary ("...we use Kafka heavily here, so I'm wondering how you'd guarantee ordering across partitions, anyway that's the setup.") — extract the embedded question.
   Only genuine statements with NO expectation of a reply (acknowledgements, the interviewer describing the company, filler) are is_question=false.
2. question: if is_question, return the question with only LIGHT cleanup — fix punctuation/capitalization, drop filler ("um", "so", "like") and false starts, and resolve pronouns against the recent questions (e.g. expand "and how does that scale?"). 
   - KEEP THE INTERVIEWER'S WORDS. Do NOT swap a word for a different one unless it is clearly not a real English/technical word AND a phonetically near-identical term is the obvious fix.
   - A real word (e.g. "cohesion", "innovation", "binding") is almost always what was said — do NOT replace it with a different technical term just because that term is common in interviews.
   - DOMAIN REPAIR (only when a DOMAIN block is given above): STT often mishears technical terms as ordinary words or wrong tokens. When a transcript word is phonetically close to — or an obvious component/variant of — the interview's DOMAIN, correct it to the intended term. Examples given a Java/Kubernetes/Docker domain: "spring" in a coding question about a string -> "string"; "Q proxy" / "cube proxy" -> "kube-proxy"; "cube cuttle" / "cube CTL" -> "kubectl"; "post grease" -> "PostgreSQL"; "doctor file" -> "Dockerfile". Make the correction ONLY when the domain makes the intended term unambiguous; otherwise keep the word.
   - Outside a domain match, NEVER introduce a named model, person, acronym, framework, or proper noun that is not phonetically present in the transcript. (E.g. never turn "cohesion" into "Boehm's model".)
   - When in doubt, return the transcript verbatim (cleaned of filler only).
   Empty string if not a question.
3. type: one of "coding", "technical_concept", "behavioral", "clarification", "smalltalk".
4. topic: a 1-3 word topic tag.
5. difficulty: how hard the question is to answer well — one of "trivial", "standard", "hard", "expert".

Return ONLY a JSON object:
{{"is_question": true, "question": "...", "type": "technical_concept", "topic": "...", "difficulty": "standard"}}"""

def _model() -> str | None:
    """Use the fastest available model for prediction so the live path stays
    snappy: the pinned live_model (a fast inference provider) first, then the
    classifier model, else let the router pick. None lets the router decide."""
    return getattr(cfg.llm, "live_model", None) or cfg.llm.classifier_model or None

def _heuristic_fallback(text: str) -> Prediction:
    """Deterministic fallback when the LLM path is down/stalled/garbled: the
    heuristic classifier decides instead of assuming everything is a question —
    otherwise explanations and small talk all get answered while the model is
    degraded. The transcript passes through unchanged."""
    from app.question_detection.classifier import heuristic_classify
    try:
        meta = heuristic_classify(text)
    except Exception:  # noqa: BLE001 — last resort: don't drop a real question
        return Prediction(True, text, "technical_concept", "")
    qtype = meta.type if meta.type in _VALID_TYPES else "technical_concept"
    return Prediction(meta.is_question, text if meta.is_question else "",
                      qtype, meta.topic)

async def predict(transcript: str, recent: list[str],
                  *, domain: str = "") -> Prediction:
    """Predict the question from a transcript via a fast LLM. Never raises —
    on any failure the HEURISTIC classifier decides (question-or-not) and the
    transcript passes through unchanged. `domain` is an optional context block
    (resume skills / target role / topics) that unlocks confident repair of
    mis-transcribed technical terms (see the prompt's DOMAIN REPAIR rule)."""
    text = (transcript or "").strip()
    if not text:
        return Prediction(False, "", "smalltalk", "")

    recent_block = "\n".join(f"- {q}" for q in recent[-3:]) or "(none)"
    prompt = fill(_PROMPT, recent=recent_block, domain=(domain or "").strip(),
                  transcript=text.replace('"', "'"))
    try:
        # Bound the prediction so a stalled/rate-limited classifier model can't
        # hang the transcript→meta step.
        raw = await asyncio.wait_for(
            llm.chat_json([{"role": "user", "content": prompt}], model=_model()),
            timeout=15.0,
        )
    except (LLMError, asyncio.TimeoutError):
        return _heuristic_fallback(text)

    parsed = _parse_lenient_json(raw)
    if not isinstance(parsed, dict):
        return _heuristic_fallback(text)

    is_q = bool(parsed.get("is_question", True))
    question = (str(parsed.get("question") or "").strip()) or text
    qtype = parsed.get("type") or "technical_concept"
    if qtype not in _VALID_TYPES:
        qtype = "technical_concept"
    topic = str(parsed.get("topic") or "").strip()[:60]
    difficulty = str(parsed.get("difficulty") or "standard").strip().lower()
    if difficulty not in _VALID_DIFFICULTY:
        difficulty = "standard"
    return Prediction(is_q, question if is_q else "", qtype, topic, difficulty)
