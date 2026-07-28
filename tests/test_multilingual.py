"""Tests for multilingual voice + code-switching (vNext §10.6.5, Stage 10 F)."""
from __future__ import annotations

import app.live.multilingual as M


def _on(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)


# ---- per-stage capability contract ---------------------------------------
def test_stt_final_native_everywhere(monkeypatch):
    _on(monkeypatch)
    for lang in ("en", "hi", "ja", "sw", "xx"):
        c = M.plan_stage(M.STT_FINAL, lang)
        assert c.mode == M.NATIVE and not c.degraded    # Whisper turbo backbone


def test_stt_partial_english_native_pack_or_degrade(monkeypatch):
    _on(monkeypatch)
    assert M.plan_stage(M.STT_PARTIAL, "en").mode == M.NATIVE
    assert M.plan_stage(M.STT_PARTIAL, "es").mode == M.FAST_PACK
    d = M.plan_stage(M.STT_PARTIAL, "hi")
    assert d.mode == M.DEGRADE and d.degraded         # turbo-only partials


def test_llm_always_native(monkeypatch):
    _on(monkeypatch)
    for lang in ("en", "hi", "zz"):
        assert M.plan_stage(M.LLM, lang).mode == M.NATIVE


def test_tts_voice_pack_or_cloud(monkeypatch):
    _on(monkeypatch)
    assert M.plan_stage(M.TTS, "hi").mode == M.NATIVE          # has a pack
    c = M.plan_stage(M.TTS, "sw")
    assert c.mode == M.CLOUD_FALLBACK and c.degraded          # no pack → cloud


def test_wake_pack_or_push_to_talk(monkeypatch):
    _on(monkeypatch)
    assert M.plan_stage(M.WAKE, "en").mode == M.NATIVE
    c = M.plan_stage(M.WAKE, "ja")
    assert c.mode == M.PUSH_TO_TALK and c.degraded            # universal fallback


def test_session_plan_covers_all_stages(monkeypatch):
    _on(monkeypatch)
    plan = M.session_plan("hi")
    assert set(plan) == {M.STT_FINAL, M.STT_PARTIAL, M.LLM, M.TTS, M.WAKE}


def test_disabled_is_english_native(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: False)
    c = M.plan_stage(M.TTS, "hi")
    assert c.mode == M.NATIVE and c.language == "en"          # byte-identical


def test_plan_never_raises(monkeypatch):
    _on(monkeypatch)
    assert isinstance(M.plan_stage(M.WAKE, None), M.StageCapability)  # type: ignore[arg-type]


# ---- cross-lingual emotion adjustment ------------------------------------
def test_english_valence_full_weight(monkeypatch):
    _on(monkeypatch)
    a = M.adjust_emotion(0.8, 0.6, language="en")
    assert a.valence_weight == 1.0 and a.valence == 0.8


def test_arousal_transfers_unchanged(monkeypatch):
    _on(monkeypatch)
    for lang in ("en", "hi", "ja"):
        assert M.adjust_emotion(0.5, 0.7, language=lang).arousal == 0.7


def test_non_english_valence_downweighted(monkeypatch):
    _on(monkeypatch)
    a = M.adjust_emotion(0.8, 0.6, language="es")
    assert a.valence_weight < 1.0 and abs(a.valence) < 0.8


def test_high_context_language_downweighted_more(monkeypatch):
    _on(monkeypatch)
    es = M.adjust_emotion(0.8, 0.6, language="es").valence_weight
    ja = M.adjust_emotion(0.8, 0.6, language="ja").valence_weight
    assert ja < es                                   # high-context → lower weight


def test_low_confidence_downweights_valence(monkeypatch):
    _on(monkeypatch)
    hi = M.adjust_emotion(0.8, 0.6, language="hi", confidence=1.0).valence_weight
    lo = M.adjust_emotion(0.8, 0.6, language="hi", confidence=0.5).valence_weight
    assert lo < hi


def test_adjust_never_raises(monkeypatch):
    _on(monkeypatch)
    assert isinstance(M.adjust_emotion(None, None, language="en"),  # type: ignore[arg-type]
                      M.EmotionAdjustment)


# ---- code-switch detection ------------------------------------------------
def test_codeswitch_detects_mixing_and_tech_terms():
    r = M.detect_codeswitch("मैंने Kafka use किया, then kubectl", session_language="hi")
    assert r.mixed and r.primary == "hi"
    assert "Kafka" in r.tech_terms and "kubectl" in r.tech_terms


def test_pure_english_not_mixed():
    r = M.detect_codeswitch("what is the difference between redis and postgres")
    assert not r.mixed
    assert {"redis", "postgres"}.issubset(set(r.tech_terms))


def test_tech_terms_deduped():
    r = M.detect_codeswitch("kafka kafka KAFKA and docker")
    lows = [t.lower() for t in r.tech_terms]
    assert lows.count("kafka") == 1


def test_detect_never_raises():
    assert isinstance(M.detect_codeswitch(None), M.CodeSwitchRead)  # type: ignore[arg-type]


def test_answer_directive_keeps_tech_terms():
    d = M.answer_language_directive("hi", tech_terms=["Kafka", "kubectl"])
    assert "hi" in d and "Kafka" in d and "kubectl" in d


def test_answer_directive_english_no_terms_is_blank():
    assert M.answer_language_directive("en") == ""


def test_answer_directive_english_with_terms():
    d = M.answer_language_directive("en", tech_terms=["Kafka"])
    assert "Kafka" in d                              # keep terms even in English


# --- B1: per-turn answer language incl. Indian scripts (app/live/language.py) --
import app.live.language as L


def test_indian_scripts_detected_per_turn():
    """Telugu/Tamil/Kannada/Hindi questions detect their own language (was: only
    Devanagari mapped, so Telugu etc. silently degraded to English)."""
    cases = {
        "కాఫ్కా ఎలా పని చేస్తుంది": "te",
        "காஃப்கா எப்படி வேலை செய்கிறது": "ta",
        "ಕಾಫ್ಕಾ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ": "kn",
        "काफ्का कैसे काम करता है": "hi",
    }
    for text, code in cases.items():
        assert L.detect_language(text) == code, text
        # auto answer_language → a directive to reply in that language.
        assert L.answer_directive(L.target_language(L.detect_language(text)))


def test_english_question_no_language_directive():
    assert L.detect_language("How does Kafka handle ordering") == "en"
    assert L.answer_directive(L.target_language("en")) == ""
