"""Multilingual voice + code-switching (vNext §10.6.5, Stage 10 Component F).

Every voice STAGE has a different language reach; the contract is: degrade
GRACEFULLY, never hard-fail. Per stage:

  * **STT finals** — Whisper-turbo (~99 langs) is the backbone → native everywhere;
  * **STT partials** — Parakeet is English-centric → a fast-pack where one exists,
    else DEGRADE to turbo-only partials (same finals, slightly higher latency);
  * **LLM answer** — always the session language (a prompt directive);
  * **TTS** — per-language voice packs, else a CLOUD-TTS fallback;
  * **wake word** — per-language packs, else PUSH-TO-TALK as the universal
    day-one fallback;
  * **emotion** — prosody/arousal is language-independent (transfers), valence is
    culturally coded → down-weight low-confidence valence for non-English.

**Code-switching** (intra-sentence mixing, "मैंने Kafka use किया, then the consumer
lagged") is expected, not broken: STT declares a PRIMARY language without
hard-filtering, so embedded English tech terms transcribe correctly; the answer
honors the session-language prose but keeps technical terms as-is.

This module owns the per-stage capability CONTRACT, the cross-lingual emotion
adjustment, and code-switch detection; the engines/packs are the injected seams.
Pure + fail-open. Flag-gated (`voice.multilingual`, default OFF → English-only
assumptions, no degradation plan).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Voice stages.
STT_FINAL = "stt_final"
STT_PARTIAL = "stt_partial"
LLM = "llm"
TTS = "tts"
WAKE = "wake"

# Capability modes (how a stage serves a language).
NATIVE = "native"
FAST_PACK = "fast_pack"
DEGRADE = "degrade"           # works, reduced (e.g. turbo-only partials)
CLOUD_FALLBACK = "cloud_fallback"
PUSH_TO_TALK = "push_to_talk"

# Per-stage coverage (the real packs load on the GPU plane; this is the contract).
_STT_PARTIAL_PACKS = {"en", "es", "fr", "de", "it", "pt"}   # Parakeet fast packs
_TTS_VOICE_PACKS = {"en", "hi", "es", "fr", "de", "ja", "zh"}
_WAKE_PACKS = {"en", "es", "hi"}
# Whisper-turbo backbone — effectively all; a tiny "unsupported" set degrades.
_STT_FINAL_UNSUPPORTED: set = set()
# Languages whose emotional VALENCE is most culturally coded (down-weight more).
_HIGH_CONTEXT_LANGS = {"ja", "zh", "ko", "hi", "th", "vi"}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.voice, "multilingual", False))
    except Exception:  # noqa: BLE001
        return False


def _lang(language: str) -> str:
    return (language or "en").strip().lower()[:2] or "en"


# --------------------------------------------------------------------------- #
# Per-stage capability contract
# --------------------------------------------------------------------------- #
@dataclass
class StageCapability:
    stage: str
    language: str
    mode: str
    degraded: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": self.stage, "language": self.language, "mode": self.mode,
                "degraded": self.degraded, "note": self.note}


def plan_stage(stage: str, language: str) -> StageCapability:
    """The per-stage language capability + graceful-degrade plan. Disabled →
    NATIVE English assumption (byte-identical). Never raises → a safe degrade."""
    try:
        lang = _lang(language)
        if not enabled():
            return StageCapability(stage, "en", NATIVE, False, "disabled")
        if stage == STT_FINAL:
            if lang in _STT_FINAL_UNSUPPORTED:
                return StageCapability(stage, lang, DEGRADE, True,
                                       "outside Whisper coverage — best-effort")
            return StageCapability(stage, lang, NATIVE, False, "Whisper turbo")
        if stage == STT_PARTIAL:
            if lang == "en":
                return StageCapability(stage, lang, NATIVE, False, "Parakeet")
            if lang in _STT_PARTIAL_PACKS:
                return StageCapability(stage, lang, FAST_PACK, False,
                                       "per-language partial pack")
            return StageCapability(stage, lang, DEGRADE, True,
                                   "turbo-only partials (same finals, +latency)")
        if stage == LLM:
            return StageCapability(stage, lang, NATIVE, False,
                                   "session-language prompt directive")
        if stage == TTS:
            if lang in _TTS_VOICE_PACKS:
                return StageCapability(stage, lang, NATIVE, False, "voice pack")
            return StageCapability(stage, lang, CLOUD_FALLBACK, True,
                                   "no local pack — cloud TTS")
        if stage == WAKE:
            if lang in _WAKE_PACKS:
                return StageCapability(stage, lang, NATIVE, False, "wake pack")
            return StageCapability(stage, lang, PUSH_TO_TALK, True,
                                   "no wake pack — push-to-talk")
        return StageCapability(stage, lang, DEGRADE, True, "unknown stage")
    except Exception:  # noqa: BLE001
        return StageCapability(stage, "en", DEGRADE, True, "error → degrade")


def session_plan(language: str) -> dict:
    """The full per-stage contract for a session language (for the setup UI +
    routing). Never raises."""
    return {s: plan_stage(s, language).to_dict()
            for s in (STT_FINAL, STT_PARTIAL, LLM, TTS, WAKE)}


# --------------------------------------------------------------------------- #
# Cross-lingual emotion adjustment
# --------------------------------------------------------------------------- #
@dataclass
class EmotionAdjustment:
    valence: float
    arousal: float
    valence_weight: float         # how much to trust the valence read (0..1)
    note: str = ""

    def to_dict(self) -> dict:
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "valence_weight": round(self.valence_weight, 3), "note": self.note}


def adjust_emotion(valence: float, arousal: float, *, language: str,
                   confidence: float = 1.0) -> EmotionAdjustment:
    """Adjust an emotion read for the session language. Arousal is language-
    INDEPENDENT (a raised, fast voice reads as activated everywhere) → unchanged;
    VALENCE is culturally coded → its weight drops for high-context languages and
    for low-confidence reads, so the fusion leans on arousal + transcript
    sentiment instead. Never raises → an unweighted pass-through."""
    try:
        lang = _lang(language)
        weight = 1.0
        if lang != "en":
            weight *= 0.75                       # non-English valence less certain
        if lang in _HIGH_CONTEXT_LANGS:
            weight *= 0.7                        # high-context → down-weight more
        weight *= max(0.0, min(1.0, confidence))
        return EmotionAdjustment(
            valence=valence * weight, arousal=arousal,   # arousal transfers as-is
            valence_weight=weight,
            note=("valence down-weighted (culturally coded)" if weight < 1.0
                  else "full-weight valence"))
    except Exception:  # noqa: BLE001
        return EmotionAdjustment(valence, arousal, 1.0, "error → pass-through")


# --------------------------------------------------------------------------- #
# Code-switch detection
# --------------------------------------------------------------------------- #
# Common technical terms/tokens that must survive canonicalization as-is.
_TECH_RE = re.compile(
    r"\b(?:kafka|kubectl|kubernetes|docker|redis|postgres(?:ql)?|nginx|"
    r"grpc|graphql|api|sql|json|yaml|http[s]?|tcp|udp|ssl|jwt|oauth|"
    r"react|vue|angular|node|npm|git|regex|async|await|lambda|"
    r"[A-Za-z_][A-Za-z0-9_]*\(\)|[a-z]+\.[a-z]+\([^)]*\))\b", re.IGNORECASE)
# Latin-script run detector (for mixed-script code-switch).
_LATIN = re.compile(r"[A-Za-z]{2,}")
_NON_LATIN = re.compile(r"[^\x00-\x7F]{2,}")


@dataclass
class CodeSwitchRead:
    primary: str                  # declared primary language (best-effort)
    mixed: bool                   # intra-sentence language mixing detected
    tech_terms: list = field(default_factory=list)  # terms to preserve as-is

    def to_dict(self) -> dict:
        return {"primary": self.primary, "mixed": self.mixed,
                "tech_terms": list(self.tech_terms)}


def detect_codeswitch(text: str, *, session_language: str = "en") -> CodeSwitchRead:
    """Declare a PRIMARY language without hard-filtering, flag intra-sentence
    mixing, and extract the technical terms that must be preserved verbatim. The
    session language is the primary default (STT already declared it); mixing is
    True when both a non-Latin run and a Latin run co-occur. Never raises."""
    try:
        t = text or ""
        tech = []
        seen = set()
        for m in _TECH_RE.finditer(t):
            term = m.group(0)
            key = term.lower()
            if key not in seen:
                seen.add(key)
                tech.append(term)
        has_latin = bool(_LATIN.search(t))
        has_non_latin = bool(_NON_LATIN.search(t))
        mixed = has_latin and has_non_latin
        primary = _lang(session_language)
        return CodeSwitchRead(primary=primary, mixed=mixed, tech_terms=tech)
    except Exception:  # noqa: BLE001
        return CodeSwitchRead(_lang(session_language), False, [])


def answer_language_directive(session_language: str, *, tech_terms=None) -> str:
    """The prompt directive: answer in the session language's prose but keep
    technical terms conventional (as-is). '' when English + no terms."""
    try:
        lang = _lang(session_language)
        terms = list(tech_terms or [])
        if lang == "en" and not terms:
            return ""
        base = (f"Answer in {lang}." if lang != "en" else "")
        if terms:
            base += (" Keep technical terms in their conventional form: "
                     + ", ".join(terms[:12]) + ".")
        return base.strip()
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["STT_FINAL", "STT_PARTIAL", "LLM", "TTS", "WAKE", "NATIVE",
           "FAST_PACK", "DEGRADE", "CLOUD_FALLBACK", "PUSH_TO_TALK", "enabled",
           "StageCapability", "plan_stage", "session_plan", "EmotionAdjustment",
           "adjust_emotion", "CodeSwitchRead", "detect_codeswitch",
           "answer_language_directive"]
