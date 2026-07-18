"""Classify a turn's computational difficulty → drives capability-aware routing
and a rigor directive.

LLM-driven (one fast JSON call, no keyword rules). The label flows into
`llm.stream_chat(..., options={"difficulty": label})`, where the router weights
model intelligence accordingly — hard/expert work goes to the strongest
available model — and a matching rigor directive is appended to the system
prompt to push toward correct, elegant solutions.
"""
from __future__ import annotations

import json
import logging
import re

from app.core import lexicons

log = logging.getLogger(__name__)

# Difficulty ladder — typed named constants (single source of truth). Use these
# instead of raw string literals so a typo is an ImportError, not silent
# mis-routing (audit §17b). `LEVELS` is derived from them and unchanged.
TRIVIAL = "trivial"
STANDARD = "standard"
HARD = "hard"
EXPERT = "expert"
LEVELS = (TRIVIAL, STANDARD, HARD, EXPERT)


def is_level(value: str) -> bool:
    """True if `value` is a valid difficulty level (for validating overrides)."""
    return isinstance(value, str) and value.strip().lower() in LEVELS

# Lexicon DATA lives in the central registry (app/core/lexicons.py); this
# module only compiles it.
# Greetings / acknowledgements that are obviously trivial — short-circuited
# WITHOUT an LLM round-trip so simple chat stays instant.
_TRIVIAL_PHRASES = lexicons.DIFFICULTY_TRIVIAL_PHRASES

# Heavy/large-scope generation signals → route IMMEDIATELY to the strongest
# model ("expert") without spending an LLM round-trip on classification.
_HEAVY_RE = re.compile(lexicons.DIFFICULTY_HEAVY, re.IGNORECASE | re.DOTALL)
# Explicit magnitude: "1000 files", "100000 lines", "50 components/endpoints…".
_MAGNITUDE_RE = re.compile(lexicons.DIFFICULTY_MAGNITUDE, re.IGNORECASE)


# Explanation/overview cues — these are NOT heavy generation even when they
# mention a big scope ("explain the whole project"), so they veto the heuristic
# and route by normal difficulty instead of the slow top-tier model.
_EXPLAIN_RE = re.compile(lexicons.DIFFICULTY_EXPLAIN, re.IGNORECASE)


def _is_heavy(text: str) -> bool:
    """Deterministic 'huge task' detector for immediate top-tier routing.

    Heavy = a large GENERATION/build task, not merely any mention of a big
    scope. An explanation/overview request ("explain the whole project",
    "summarize the full app") must NOT be escalated to the slow giant — so an
    explanation cue vetoes the heuristic.
    """
    if _EXPLAIN_RE.search(text):
        return False
    if _MAGNITUDE_RE.search(text):
        return True
    return bool(_HEAVY_RE.search(text))


# Build-request detection shared by the chat-mesh and upload paths: an
# ambiguous build request is "build/create an open-ended PROJECT/app" with NO
# language/framework named anywhere in the recent window → the Clarifier should
# ask (which language/framework/platform) before answering. A self-contained
# code task ("write a program to reverse a string in Java") is NOT this — it is
# directly answerable and must never trigger a forced clarification.
# Open-ended PROJECT nouns — these denote a multi-file deliverable whose
# language/framework/platform genuinely change everything, so a build request
# naming one of these with NO tech is worth one clarification. Deliberately
# EXCLUDES program/script/function/snippet (those are single-file code answers).
_PROJECT_NOUN_RE = re.compile(lexicons.DIFFICULTY_PROJECT_NOUN, re.IGNORECASE)
# Verbs that genuinely start a project build (not "write/give me/need" which
# usually precede a one-off snippet).
_PROJECT_VERB_RE = re.compile(lexicons.DIFFICULTY_PROJECT_VERB, re.IGNORECASE)
_TECH_RE = re.compile(lexicons.DIFFICULTY_TECH, re.IGNORECASE)


def is_ambiguous_build_request(current: str, recent: str = "") -> bool:
    """True ONLY when the user asks to build an open-ended PROJECT/app and named
    no language/framework anywhere in the recent window — the cue to clarify.

    A self-contained code task ("a program to reverse a string in Java using
    streams") is directly answerable and is NOT an ambiguous build, so it
    returns False even though it contains a build verb + the noun 'program'.
    Requiring a PROJECT noun (app/website/system/…, not program/script) plus a
    project verb (build/create/…, not write/give me) is what excludes snippets.
    """
    cur = (current or "").lower()
    if not (_PROJECT_VERB_RE.search(cur) and _PROJECT_NOUN_RE.search(cur)):
        return False
    return _TECH_RE.search(f"{recent} {current}".lower()) is None


def is_build_request(current: str) -> bool:
    """True when the user asks to BUILD/create a project/app/etc. (regardless of
    whether a language/framework was named) — the cue to demand COMPLETE,
    layout-consistent file generation so the project is downloadable. Scoped to
    PROJECT nouns + project verbs so a one-off 'write a program to …' snippet is
    not forced through the heavyweight whole-project generation path."""
    cur = (current or "").lower()
    return bool(_PROJECT_VERB_RE.search(cur) and _PROJECT_NOUN_RE.search(cur))

_PROMPT = (
    "Rate how computationally demanding it is to answer the user's message WELL "
    "(not how long the answer is). Use exactly one label:\n"
    "- trivial: greetings, small talk, one-line facts, trivial edits.\n"
    "- standard: ordinary questions, explanations, straightforward code.\n"
    "- hard: multi-step reasoning, non-trivial algorithms/math, debugging gnarly "
    "code, system design, careful analysis of a document/codebase.\n"
    "- expert: deep or novel problem-solving — proofs, intricate optimization, "
    "subtle concurrency/security, multi-constraint design, research-level work.\n\n"
    "Reply with ONLY compact JSON: {\"difficulty\": \"trivial|standard|hard|expert\"}\n\n"
    "User message:\n{text}"
)

# Appended to the system prompt for demanding turns — nudges rigor + elegance.
_RIGOR = (
    "\n\nThis is a demanding task. Be maximally rigorous: reason it through "
    "carefully and completely before answering, check every step, computation, "
    "and edge case, and prefer a correct, clean, elegant solution over a quick "
    "one. For anything quantitative or algorithmic, verify the result (and state "
    "complexity where relevant). If several approaches exist, pick the clearest "
    "correct one. Do not hand-wave or guess — if something is genuinely "
    "uncertain, say so explicitly."
)


def rigor_directive(difficulty: str) -> str:
    """System-prompt addendum for hard/expert turns (empty otherwise)."""
    return _RIGOR if difficulty in ("hard", "expert") else ""


async def classify_difficulty(text: str, recent: str = "") -> str:
    """Return one of LEVELS. LLM-driven; safe default 'standard' on empty input
    or any failure. `recent` is a short transcript of the last turn(s) so a
    follow-up ("now make it lock-free", "optimize that") is rated in context,
    not in isolation."""
    t = (text or "").strip()
    if not t:
        return "standard"
    # Fast path: obvious greetings / very short acks are trivial — skip the
    # classifier LLM call entirely so simple turns respond instantly. (Kept
    # tight: a 3-4 char acronym like "DFS"/"OOP?" is NOT trivial.)
    low = t.lower().strip(" \t\n!.?,;:")
    if low in _TRIVIAL_PHRASES or len(low) <= 2:
        return "trivial"
    # Fast path: an obviously HEAVY/large-scope task goes straight to the
    # strongest model — no need to ask the classifier.
    if _is_heavy(t):
        return "expert"
    try:
        from app.core.config_loader import cfg
        from app.core.llm_client import LLMError, llm
        from app.core.prompt import fill

        prompt = _PROMPT
        ctx = (recent or "").strip()
        if ctx:
            prompt = (
                "Recent conversation (for context — the latest message may be a "
                "follow-up that inherits this difficulty):\n" + ctx[:2000]
                + "\n\n" + _PROMPT
            )
        raw = await llm.complete(
            [{"role": "user", "content": fill(prompt, text=t[:4000])}],
            model=(cfg.llm.classifier_model or cfg.llm.model),
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.micro_label},
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001
        log.info("difficulty classify failed (default standard): %s", exc)
        return "standard"
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        lvl = str(json.loads(s).get("difficulty", "")).lower().strip()
    except Exception:  # noqa: BLE001
        lvl = next((w for w in LEVELS if w in (raw or "").lower()), "")
    return lvl if lvl in LEVELS else "standard"


__all__ = ["classify_difficulty", "rigor_directive", "LEVELS",
           "is_ambiguous_build_request", "is_build_request"]
