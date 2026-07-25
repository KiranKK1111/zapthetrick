"""TTS lane — spoken transform + sentence pipelining (vNext §10.5, Stage 10 A).

`tts.py::speech_markup` already turns Markdown into speech-ready text, but for a
real voice reply two things matter more:

  * **spoken transform** — a code block should not be read aloud char-by-char (or
    dropped silently); it becomes "I've put the code on screen." Tables, diagrams,
    and images get the same "on screen" treatment, so the spoken answer stays
    natural while the visual detail lives in the transcript.
  * **sentence pipelining** — synthesis must start on the FIRST sentence (first
    audio < 300 ms), not wait for the whole answer. `sentence_chunks` splits the
    spoken text into speakable units without breaking on abbreviations/decimals.

Plus per-user **voice selection** (a voice pack by id/language). The Kokoro-class
synthesis itself is the on-GPU seam; this owns the deterministic text pipeline.
Pure + fail-open. Flag-gated (`voice.tts`, default OFF → today's text-only reply).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Abbreviations that must NOT trigger a sentence split (the "." is not a period).
_NON_TERMINAL = (
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "mr.", "mrs.", "ms.", "dr.",
    "st.", "fig.", "no.", "vol.", "inc.", "ltd.", "u.s.", "a.m.", "p.m.",
)
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_IMG = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MERMAID = re.compile(r"```mermaid.*?```", re.DOTALL | re.IGNORECASE)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.voice, "tts", False))
    except Exception:  # noqa: BLE001
        return False


def to_spoken(markdown: str) -> str:
    """Turn a Markdown answer into natural SPOKEN text. Unlike `speech_markup`,
    visual content is announced ("on screen") rather than dropped or read. Never
    raises → a best-effort strip."""
    try:
        t = markdown or ""
        # Visual content → an "on screen" announcement (order matters: mermaid
        # before generic fences; images before link-stripping).
        t = _MERMAID.sub(" I've put a diagram on screen. ", t)
        t = _FENCE.sub(" I've put the code on screen. ", t)
        t = _IMG.sub(r" (there's an image on screen: \1) ", t)
        if _TABLE_ROW.search(t):
            t = _TABLE_ROW.sub("", t)
            t = t.rstrip() + " There's a table on screen. "
        # Inline code → its text (short, speakable).
        t = re.sub(r"`([^`]*)`", r"\1", t)
        # Links → link text.
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        # Headings / emphasis / blockquote markers.
        t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"[*_]{1,3}", "", t)
        t = re.sub(r"^\s*>\s?", "", t, flags=re.MULTILINE)
        # List markers → sentence flow.
        t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)
        # Expand a few abbreviations that read badly.
        for k, v in {"e.g.": "for example", "i.e.": "that is",
                     "etc.": "and so on", "vs.": "versus"}.items():
            t = t.replace(k, v)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:  # noqa: BLE001
        return (markdown or "").strip()


# Private-use sentinel standing in for a protected "." during sentence splitting.
_DOT = ""


def _protect_abbrev(text: str) -> str:
    out = text
    for a in _NON_TERMINAL:
        out = re.sub(re.escape(a), a.replace(".", _DOT), out, flags=re.IGNORECASE)
    # Protect decimals ("3.14") and version dots ("v2.0").
    out = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + _DOT + m.group(2), out)
    return out


def _restore_abbrev(text: str) -> str:
    return text.replace(_DOT, ".")


def sentence_chunks(spoken: str, *, max_chars: int = 240) -> "list[str]":
    """Split SPOKEN text into speakable sentence chunks — synthesis starts on the
    first chunk (first-audio latency). Never splits inside an abbreviation or a
    decimal; a very long sentence is soft-wrapped at `max_chars` on a clause
    boundary. Never raises → [the whole text] on error."""
    try:
        s = (spoken or "").strip()
        if not s:
            return []
        protected = _protect_abbrev(s)
        parts = [p.strip() for p in _SENT_END.split(protected) if p.strip()]
        chunks: list[str] = []
        for p in parts:
            p = _restore_abbrev(p)
            if len(p) <= max_chars:
                chunks.append(p)
            else:
                chunks.extend(_soft_wrap(p, max_chars))
        return chunks or [s]
    except Exception:  # noqa: BLE001
        return [(spoken or "").strip()] if (spoken or "").strip() else []


def _soft_wrap(text: str, max_chars: int) -> "list[str]":
    """Break a long sentence at clause boundaries (commas/semicolons), else words."""
    out: list[str] = []
    cur = ""
    for token in re.split(r"(,|;|:)", text):
        if len(cur) + len(token) > max_chars and cur:
            out.append(cur.strip())
            cur = token
        else:
            cur += token
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


def first_chunk(spoken: str) -> str:
    """The first speakable chunk — the one that drives first-audio latency."""
    chunks = sentence_chunks(spoken)
    return chunks[0] if chunks else ""


# --------------------------------------------------------------------------- #
# Voice selection
# --------------------------------------------------------------------------- #
@dataclass
class Voice:
    id: str
    name: str
    language: str = "en"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "language": self.language}


# A small default voice registry (the real packs load on the GPU plane).
VOICES: dict[str, Voice] = {
    "en_warm": Voice("en_warm", "Warm (EN)", "en"),
    "en_crisp": Voice("en_crisp", "Crisp (EN)", "en"),
    "hi_warm": Voice("hi_warm", "Warm (HI)", "hi"),
    "es_warm": Voice("es_warm", "Warm (ES)", "es"),
}
_DEFAULT_BY_LANG = {"en": "en_warm", "hi": "hi_warm", "es": "es_warm"}


def resolve_voice(user_pref: str = "", *, language: str = "en") -> Voice:
    """Pick a voice: an explicit user pref (if it exists AND matches the language,
    or the language is unspecified) wins; else the language default; else the
    English default. Never raises."""
    try:
        lang = (language or "en").strip().lower()
        pref = (user_pref or "").strip()
        v = VOICES.get(pref)
        if v is not None and (not language or v.language == lang):
            return v
        default_id = _DEFAULT_BY_LANG.get(lang, "en_warm")
        return VOICES.get(default_id, VOICES["en_warm"])
    except Exception:  # noqa: BLE001
        return VOICES["en_warm"]


def voices_for_language(language: str) -> "list[Voice]":
    lang = (language or "en").strip().lower()
    return [v for v in VOICES.values() if v.language == lang]


__all__ = ["enabled", "to_spoken", "sentence_chunks", "first_chunk", "Voice",
           "VOICES", "resolve_voice", "voices_for_language"]
