"""Tests for conversational intelligence — turn-taking + barge-in (§10.6.2, Stage 10 C)."""
from __future__ import annotations

import app.live.turn_taking as T


def _on(monkeypatch):
    monkeypatch.setattr(T, "enabled", lambda: True)


# ---- adaptive endpointing -------------------------------------------------
def test_complete_falling_answers_fast(monkeypatch):
    _on(monkeypatch)
    d = T.endpoint_decision(silence_ms=400, completeness=0.9, pitch_contour="falling")
    assert d.window_ms == 300 and d.endpoint and d.complete


def test_incomplete_rising_waits_longer(monkeypatch):
    _on(monkeypatch)
    d = T.endpoint_decision(silence_ms=400, completeness=0.2, pitch_contour="rising")
    assert d.window_ms == 1500 and not d.endpoint and not d.complete


def test_window_clamped_to_range(monkeypatch):
    _on(monkeypatch)
    assert T.endpoint_decision(silence_ms=0, completeness=1.0,
                               pitch_contour="falling").window_ms == 300
    assert T.endpoint_decision(silence_ms=0, completeness=0.0,
                               pitch_contour="rising").window_ms == 1500


def test_endpoint_true_when_silence_meets_window(monkeypatch):
    _on(monkeypatch)
    d = T.endpoint_decision(silence_ms=1000, completeness=0.5, pitch_contour="flat")
    assert d.window_ms == 900 and d.endpoint


def test_rising_pitch_is_incomplete_even_if_words_complete(monkeypatch):
    _on(monkeypatch)
    d = T.endpoint_decision(silence_ms=2000, completeness=0.9, pitch_contour="rising")
    assert not d.complete                         # rising → still asking


def test_endpoint_disabled_fixed_window(monkeypatch):
    monkeypatch.setattr(T, "enabled", lambda: False)
    d = T.endpoint_decision(silence_ms=900, completeness=0.9, pitch_contour="falling")
    assert d.window_ms == 900                     # fixed mid, ignores prosody


def test_endpoint_never_raises(monkeypatch):
    _on(monkeypatch)
    d = T.endpoint_decision(silence_ms=100, completeness=None)  # type: ignore[arg-type]
    assert isinstance(d, T.EndpointDecision)


# ---- barge-in classifier --------------------------------------------------
def test_backchannel_does_not_stop_tts():
    for t in ("mhm", "yeah", "right", "uh-huh", "okay got it", "makes sense"):
        b = T.classify_bargein(t)
        assert b.intent == T.BACKCHANNEL and b.should_stop_tts is False, t


def test_correction_stops_tts():
    b = T.classify_bargein("no wait, actually I meant Kafka")
    assert b.intent == T.CORRECTION and b.should_stop_tts


def test_stop_stops_tts():
    for t in ("stop", "hold on", "hang on", "never mind"):
        b = T.classify_bargein(t)
        assert b.intent == T.STOP and b.should_stop_tts, t


def test_continuation_stops_tts():
    b = T.classify_bargein("so the next point I want to make is")
    assert b.intent == T.CONTINUATION and b.should_stop_tts


def test_correction_beats_backchannel_word():
    # "okay actually no" contains a backchannel word but is a correction.
    b = T.classify_bargein("okay actually no, let me restart")
    assert b.intent == T.CORRECTION


def test_semantic_classifier_injected():
    b = T.classify_bargein("whatever", classify_fn=lambda t: T.STOP)
    assert b.intent == T.STOP and b.source == "semantic"


def test_semantic_out_of_taxonomy_falls_to_cue():
    b = T.classify_bargein("mhm", classify_fn=lambda t: "banana")
    assert b.intent == T.BACKCHANNEL and b.source == "cue"


def test_semantic_error_falls_to_cue():
    b = T.classify_bargein("stop", classify_fn=lambda t: 1 / 0)
    assert b.intent == T.STOP and b.source == "cue"


def test_empty_is_continuation():
    assert T.classify_bargein("").intent == T.CONTINUATION


def test_bargein_never_raises():
    b = T.classify_bargein(None)                  # type: ignore[arg-type]
    assert isinstance(b, T.BargeInIntent)
