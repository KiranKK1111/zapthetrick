"""Wiring test — the Phase 2 completion modules are actually USED in the live WS
pipeline (imported, invoked, surfaced additively, gated + fail-open, cleaned up
on teardown). Static source assertion (no socket/audio/models), mirroring
tests/test_phase2_gaps_wiring.py.
"""
from __future__ import annotations

import pathlib

_WS = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "routes_ws.py"
_SRC = _WS.read_text(encoding="utf-8")


def test_completion_modules_imported_into_pipeline():
    for mod in ("silence", "acoustic", "interviewer_intent", "interpret",
                "cross_round", "stt_recovery", "devmode", "tts", "research",
                "multimodal", "mock"):
        assert f"from app.live import {mod}" in _SRC, f"{mod} not imported"


def test_completion_signals_invoked_and_surfaced():
    # 2A-5 silence taxonomy
    assert "_sil.classify(" in _SRC and '"silence_type"' in _SRC
    # 2A-6 acoustic adaptation adjusts confidence
    assert "_ac.assess(" in _SRC and "_ac.adjust_confidence(" in _SRC and '"acoustic"' in _SRC
    # 2A-7 code-switch
    assert "_lang2.is_code_switched(" in _SRC and '"code_switch"' in _SRC
    # 2B-9 interviewer intent
    assert "_ii.probe_intent(" in _SRC and '"interviewer_intent"' in _SRC
    # 2B-15 multi-hypothesis
    assert "_interp.interpretations(" in _SRC and '"interpretations"' in _SRC
    # 2C-18/19 evidence strength
    assert "_ev2.strength_label(" in _SRC and '"evidence_strength"' in _SRC
    # 2C-20 company style
    assert "_org2.company_style_directive(" in _SRC and '"company_style"' in _SRC
    # 2C-21 adaptive length
    assert "_ctr2.length_directive(" in _SRC and '"answer_tier"' in _SRC
    # 2C-24 cross-round memory
    assert "_cr2.record_topic(" in _SRC and "_cr0.start_round(" in _SRC and '"prior_rounds"' in _SRC
    # 2B-10 predictive pre-drafting (stash + consume)
    assert "_pred.predraft(" in _SRC and "_pred2.consume_directive(" in _SRC
    # 2D-29 dev overlay
    assert "_dev.overlay(" in _SRC and '"dev"' in _SRC
    # 2D-30 live confidence recovery
    assert "_rec.observe(" in _SRC and "_rec.should_recover(" in _SRC and "_rec.recover(" in _SRC
    # 2E-34 screen/vision multimodal
    assert "_mm.to_utterance(" in _SRC
    # 2E-35 minimal TTS speech text
    assert "_tts.speech_markup(" in _SRC and '"speech_text"' in _SRC
    # 2E-36 pre-interview research
    assert "_research.build_brief(" in _SRC
    # 2D-28 mock mode as interviewer
    assert "_mock.generate_questions(" in _SRC and '"mock_question"' in _SRC


def test_completion_signals_gated_and_failopen():
    for flag in ("silence_taxonomy", "acoustic_adaptation", "interviewer_intent",
                 "multi_hypothesis", "cross_round_memory", "predictive_drafting",
                 "dev_mode", "confidence_recovery", "voice_output",
                 "pre_interview_research"):
        assert f'getattr(cfg.live, "{flag}"' in _SRC, f"{flag} gate missing"


def test_completion_teardown_cleanup():
    assert "from app.live.stt_recovery import forget_session" in _SRC
    assert "from app.live.predict import forget_session" in _SRC


def test_preexisting_items_still_wired():
    # 2A-1 speaker intelligence (diarization -> meta.speaker)
    assert "from app.live import diarize as _dia" in _SRC
    assert '.attribute(text=utterance)' in _SRC and 'meta["speaker"]' in _SRC
    # 2A-6 prosody/emotion advisory, off the hot path
    assert "_apply_emotion(" in _SRC and '"emotion_signal"' in _SRC
    # 2D-27 delivery coaching on the candidate channel
    assert "_coaching_tips(" in _SRC and '"coaching"' in _SRC


def test_new_control_kinds_present():
    assert 'kind in ("screen_text", "multimodal")' in _SRC
    assert 'kind == "mock_start"' in _SRC
    assert 'kind == "research_brief"' in _SRC
