"""
Multilingual / code-switching support (live-conversational-intelligence R24).

`detect_language` returns the primary language of an utterance (script-based +
common-word heuristics), tolerating intra-utterance code-switching. When
multilingual support is enabled, `answer_directive` yields a one-line directive
that makes the answer come back in the detected/configured target language —
folded into the SAME generation call (no extra round trip). Deterministic +
fail-open: error/unknown → the configured default (English).
"""
from __future__ import annotations

import re

# language code -> display name (for the directive).
_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}

# Script ranges (strong signals).
_SCRIPTS = [
    ("hi", re.compile(r"[\u0900-\u097F]")),     # Devanagari
    ("ar", re.compile(r"[\u0600-\u06FF]")),     # Arabic
    ("ru", re.compile(r"[\u0400-\u04FF]")),     # Cyrillic
    ("zh", re.compile(r"[\u4E00-\u9FFF]")),     # CJK
    ("ja", re.compile(r"[\u3040-\u30FF]")),     # Hiragana/Katakana
    ("ko", re.compile(r"[\uAC00-\uD7AF]")),     # Hangul
]

# Latin-script function-word hints (weak signals). DISTINCTIVE words only — no
# single-letter articles like Portuguese " a "/" o ", which collide with the
# English article "a" and used to misfire English → Portuguese (bug: "write a
# java program that takes a number…" was answered in Portuguese).
_HINTS = {
    "es": (" el ", " la ", " los ", " las ", " una ", " porque ", " qué ",
           " cómo ", " para ", " esto ", " con "),
    "fr": (" le ", " la ", " les ", " une ", " pour ", " comment ", " avec ",
           " dans ", " vous ", " je ", " pourquoi "),
    "de": (" der ", " die ", " und ", " wie ", " warum ", " nicht ", " ist ",
           " ein ", " mit ", " das ", " ich "),
    "pt": (" que ", " porque ", " como ", " para ", " você ", " não ",
           " uma ", " isso ", " está ", " obrigado ", " com "),
}

DEFAULT_LANG = "en"


def detect_language(text: str) -> str:
    """Return the primary language code of `text`. Tolerates code-switching by
    favouring the dominant non-Latin script when present, else Latin-word hints,
    else English. Never raises."""
    t = (text or "")
    if not t.strip():
        return DEFAULT_LANG
    try:
        # Strong: non-Latin script with the most characters wins.
        best, best_n = None, 0
        for code, rx in _SCRIPTS:
            n = len(rx.findall(t))
            if n > best_n:
                best, best_n = code, n
        if best is not None and best_n >= 2:
            return best
        # Inverted punctuation is a strong Spanish signal on its own — and it
        # glues to the first word ("¿cómo"), which used to defeat the
        # space-delimited hint matching below.
        if "¿" in t or "¡" in t:
            return "es"
        # Weak: Latin function-word hints. Strip punctuation so a hint word
        # followed by a comma/question mark still matches.
        low = " " + re.sub(r"[^\w\s']", " ", t.lower()) + " "
        scores = {code: sum(low.count(h) for h in hints) for code, hints in _HINTS.items()}
        code, score = max(scores.items(), key=lambda kv: kv[1], default=(DEFAULT_LANG, 0))
        return code if score >= 2 else DEFAULT_LANG
    except Exception:  # noqa: BLE001
        return DEFAULT_LANG


def target_language(detected: str | None = None) -> str:
    """Resolve the answer's target language: an explicit config override, else
    the detected language, else the default."""
    from app.core.config_loader import cfg
    cfgd = (getattr(cfg.live, "answer_language", "auto") or "auto").strip().lower()
    if cfgd and cfgd != "auto":
        return cfgd
    return (detected or DEFAULT_LANG)


def answer_directive(lang: str) -> str:
    """A one-line directive to answer in `lang` ("" for English/unknown so the
    prompt is unchanged)."""
    code = (lang or "").strip().lower()
    if not code or code == DEFAULT_LANG:
        return ""
    name = _NAMES.get(code, code)
    return f"Respond in {name}."


# ── Per-span code-switching (roadmap Phase 2 #7 / 2A-7) ─────────────────────
# Script-run segmentation: a code-switched utterance mixes scripts INSIDE one
# sentence ("explain मुझे Kafka partitions के बारे में"), so splitting on
# punctuation alone misses it. We instead group contiguous words by their
# dominant script and label each run's language.
def _word_script(word: str) -> str:
    for code, rx in _SCRIPTS:
        if rx.search(word):
            return code
    return "latin"


def code_switch_spans(text: str) -> list[tuple[str, str]]:
    """Segment an utterance into contiguous same-script runs and label each
    run's language, so intra-sentence code-switching is understood per-span
    rather than collapsed to one guess. Returns [(span_text, lang_code)]. Never
    raises → single span in the detected language."""
    t = (text or "").strip()
    if not t:
        return []
    try:
        words = t.split()
        if not words:
            return [(t, detect_language(t))]
        runs: list[tuple[list[str], str]] = []
        for w in words:
            sc = _word_script(w)
            if runs and runs[-1][1] == sc:
                runs[-1][0].append(w)
            else:
                runs.append(([w], sc))
        spans: list[tuple[str, str]] = []
        for toks, sc in runs:
            span = " ".join(toks)
            # A Latin run's language still needs the word-hint pass; a non-Latin
            # run's script IS its language.
            lang = sc if sc != "latin" else detect_language(span)
            spans.append((span, lang))
        return spans
    except Exception:  # noqa: BLE001
        return [(t, detect_language(t))]


def languages_present(text: str) -> list[str]:
    """Distinct language codes present across the utterance's spans, primary
    (dominant) first. Never raises."""
    try:
        spans = code_switch_spans(text)
        if not spans:
            return []
        counts: dict[str, int] = {}
        for span, code in spans:
            counts[code] = counts.get(code, 0) + len(span)
        return [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    except Exception:  # noqa: BLE001
        return []


def is_code_switched(text: str) -> bool:
    """True when the utterance genuinely mixes >1 language (each with real
    weight, so a lone loanword doesn't count). Never raises → False."""
    try:
        present = languages_present(text)
        return len(present) >= 2
    except Exception:  # noqa: BLE001
        return False


def code_switch_directive(text: str) -> str:
    """When the interviewer code-switches, instruct the answer to reply in the
    DOMINANT language while keeping technical terms intact. '' otherwise."""
    try:
        present = languages_present(text)
        if len(present) < 2:
            return ""
        primary = target_language(present[0])
        name = _NAMES.get(primary, primary)
        return (f"The question mixes languages — reply primarily in {name}, "
                "keeping technical terms in their original form.")
    except Exception:  # noqa: BLE001
        return ""
