"""Domain-adaptive vocabulary boosting for STT — Architecture.md.

Live transcription drops technical names ("Kafka" → "coffee",
"Postgres" → "post grass"). The doc commits to *biasing* the STT
engine with a small set of likely terms gathered from:

  - the active resume's profile + raw text
  - the user's persona / COPILOT.md ("about me")
  - the rolling window of past questions in the live session
  - a per-domain keyword pack (system design, ML, DevOps, …)

Whisper exposes this via `initial_prompt` ("a Whisper prompt with
keywords steers the recognizer"). faster-whisper has the same knob.
Parakeet uses a real keyword-boost API. This module is the
single source of truth those engines pull from.

Public surface:
    build_initial_prompt() -> str                   # short prompt with keywords
    build_boost_list(limit=120) -> list[str]        # ranked term list
    register_term(term, weight=1.0)                 # session-local boost
    add_resume_terms(resume_id)                     # async — pulls from DB
    refresh_from_persona()                          # COPILOT.md keywords

Heuristic-only — no LLM call. Cheap, deterministic, runs on every
audio chunk's startup path. ~5 ms even with a thousand terms.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from pathlib import Path

from app.core.config_loader import cfg


log = logging.getLogger(__name__)


# No hardcoded terminology pack. Domain biasing now comes ONLY from dynamic,
# context-specific sources (the candidate's resume + persona/session terms);
# cloud STT (Groq whisper-large-v3-turbo) is accurate enough on technical
# vocabulary that a baked-in term list is unnecessary. Kept empty so the
# booster degrades to "no extra bias" when there's no resume loaded.
_DEFAULT_DOMAIN_TERMS: list[str] = []


_session_terms: dict[str, float] = {}


def register_term(term: str, weight: float = 1.0) -> None:
    """Add (or bump) a single term — used by the rolling-buffer
    pane when it sees a new technical noun in the candidate's
    previous turn that the next interviewer turn might reuse."""
    key = term.strip()
    if not key or len(key) > 60:
        return
    _session_terms[key] = _session_terms.get(key, 0.0) + max(weight, 0.0)


def build_boost_list(limit: int = 120) -> list[str]:
    """Return a ranked list of terms suitable for engines with a
    real keyword-boost API (Parakeet, AssemblyAI). Highest-weight
    first; defaults are appended after the session-specific terms
    so a fresh install still benefits."""
    ranked = [t for t, _ in sorted(_session_terms.items(), key=lambda kv: -kv[1])]
    seen = set(t.lower() for t in ranked)
    for t in _DEFAULT_DOMAIN_TERMS:
        if t.lower() not in seen and len(ranked) < limit:
            ranked.append(t)
            seen.add(t.lower())
    return ranked[:limit]


def build_initial_prompt(max_chars: int = 700) -> str:
    """Compose a Whisper `initial_prompt` ≤ ~224 tokens.

    Whisper uses this as a soft bias — it doesn't constrain output,
    but tokens that appear here become more likely. Keep it concise:
    terms in a sentence cap; English; no special tokens. The broader
    bias is carried by `hotwords` (see whisper_stt); this stays small to
    respect the 224-token initial-prompt cap.
    """
    terms = build_boost_list(limit=32)
    if not terms:
        return ""
    # Single sentence, commas separating terms, ending with a period.
    text = "Topics in this interview may include: " + ", ".join(terms) + "."
    return text[:max_chars]


async def add_resume_terms(resume_id: str | uuid.UUID) -> int:
    """Walk the resume's chunks + raw_text and pull out plausible
    proper-noun-shaped terms. Adds them to the session pool.

    Returns the count of new terms added.
    """
    try:
        from storage.db import get_session_factory
        from storage.repos import ResumeRepo
    except Exception as exc:  # noqa: BLE001
        log.warning("vocabulary_boost: imports failed: %s", exc)
        return 0

    factory = get_session_factory()
    if factory is None:
        return 0
    try:
        async with factory() as db:
            repo = ResumeRepo(db)
            resume = await repo.get(resume_id)
            if resume is None:
                return 0
            text = resume.raw_text or ""
            profile = resume.profile if isinstance(resume.profile, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("vocabulary_boost: resume lookup failed: %s", exc)
        return 0

    terms = _extract_terms(text)
    # Profile skills carry domain weight — small bump so they win ties.
    for s in (profile.get("skills") or []):
        if isinstance(s, str):
            terms.append(s.strip())
        elif isinstance(s, dict) and isinstance(s.get("name"), str):
            terms.append(s["name"].strip())

    before = len(_session_terms)
    for t in terms:
        if t:
            register_term(t, weight=1.0)
    return len(_session_terms) - before


def refresh_from_persona() -> int:
    """Read COPILOT.md / persona settings and register tokens that
    look like technical names. Idempotent."""
    persona_path = _persona_file()
    if persona_path is None or not persona_path.exists():
        return 0
    try:
        body = persona_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("vocabulary_boost: COPILOT.md read failed: %s", exc)
        return 0
    before = len(_session_terms)
    for term in _extract_terms(body):
        register_term(term, weight=0.5)
    return len(_session_terms) - before


# ---- helpers -----------------------------------------------------------
_PROPER_NOUN_RE = re.compile(
    # CamelCase / hyphenated / dotted names: "Kubernetes", "Next.js",
    # "Bellman-Ford", "BigQuery"
    r"\b([A-Z][A-Za-z0-9]+(?:[-.][A-Z]?[A-Za-z0-9]+){0,3})\b"
)
_ALL_CAPS_RE = re.compile(r"\b([A-Z]{2,8})\b")
_STOP = frozenset({
    "I", "A", "AN", "THE", "AND", "OR", "BUT", "FOR", "TO", "OF", "IN",
    "ON", "AT", "BY", "WITH", "AS", "IS", "WAS", "BE", "BEEN", "IT",
    "HE", "SHE", "WE", "THEY", "THIS", "THAT", "THESE", "THOSE", "MY",
    "YOUR", "OUR", "USA", "UK", "EU", "US",
})


def _extract_terms(text: str) -> list[str]:
    """Pull proper-noun-shaped tokens out of free text. Tolerant —
    false positives are fine because they get capped by `build_boost_list`."""
    if not text:
        return []
    raw = list(_PROPER_NOUN_RE.findall(text))
    raw.extend(_ALL_CAPS_RE.findall(text))
    counts = Counter(t for t in raw if t.upper() not in _STOP and 2 <= len(t) <= 40)
    return [t for t, _ in counts.most_common(200)]


def _persona_file() -> Path | None:
    """Locate the persona file. Order: explicit cfg path, repo root, CWD."""
    candidates = [
        Path("COPILOT.md"),
        Path("backend") / "COPILOT.md",
        Path(__file__).resolve().parents[3] / "COPILOT.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def reset_for_session() -> None:
    """Drop the session-local boost pool. Called at the start of
    a new live session so terms from a prior interview don't bleed
    into the next."""
    _session_terms.clear()


__all__ = [
    "register_term",
    "build_boost_list",
    "build_initial_prompt",
    "add_resume_terms",
    "refresh_from_persona",
    "reset_for_session",
]


# Quiet flake8 — cfg is reserved for future feature flags
# (`cfg.stt.vocab_boost_enabled`).
_ = cfg
