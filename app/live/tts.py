"""
Natural voice output — minimal, honest TTS layer (roadmap Phase 2 #35 / 2E-35).

FULL streaming voice synthesis + playback is a large, largely FE/audio-device
feature and remains deferred (see the status report). What is genuinely useful
and self-contained today is the piece the backend owns: turning a Markdown
answer into clean, speech-ready text (SSML-free plain prose) and exposing
whether a local synthesis engine is available. `speak()` will drive a local
engine (pyttsx3) IFF one is installed — otherwise it returns None rather than
faking audio. The pipeline surfaces `meta.speech_text` so a client with its own
TTS (browser SpeechSynthesis, native) can voice the answer immediately.

Deterministic + fail-open. No new hard dependency.
"""
from __future__ import annotations

import re

# Common abbreviations that read badly when spoken verbatim.
_EXPAND = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "and so on",
    "vs.": "versus", "approx.": "approximately",
}


def speech_markup(text: str) -> str:
    """Convert a Markdown answer into clean speech-ready plain text: strip code
    fences, markup, list bullets and links, expand a few abbreviations, and
    collapse whitespace. Never raises → the input (stripped)."""
    try:
        t = text or ""
        # Drop fenced code blocks entirely (unspeakable).
        t = re.sub(r"```.*?```", " (code omitted) ", t, flags=re.DOTALL)
        t = re.sub(r"`([^`]*)`", r"\1", t)
        # Links -> link text.
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        # Headings / emphasis / blockquote markers.
        t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"[*_]{1,3}", "", t)
        t = re.sub(r"^\s*>\s?", "", t, flags=re.MULTILINE)
        # List bullets / numbering -> sentence flow.
        t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)
        for k, v in _EXPAND.items():
            t = t.replace(k, v)
        # Collapse whitespace.
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:  # noqa: BLE001
        return (text or "").strip()


def is_available() -> bool:
    """Whether a local synthesis engine is importable. Never raises → False."""
    try:
        import importlib.util
        return importlib.util.find_spec("pyttsx3") is not None
    except Exception:  # noqa: BLE001
        return False


def speak(text: str) -> bool:
    """Best-effort local synthesis via pyttsx3 when installed. Returns True when
    audio was produced, False otherwise (including when no engine is present —
    never fakes success). Never raises."""
    if not is_available():
        return False
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.say(speech_markup(text))
        engine.runAndWait()
        return True
    except Exception:  # noqa: BLE001
        return False


__all__ = ["speech_markup", "is_available", "speak"]
