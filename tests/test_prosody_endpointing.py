"""A5 — prosody-based endpointing: a falling-energy tail (speaker trailing off)
is a turn-final cue the segmenter uses to shave the endpoint gap. These tests
drive the real StreamingVAD energy contour deterministically by forcing every
window voiced (so the test doesn't depend on Silero classifying synthetic tones).
"""
import numpy as np
import pytest

from app.audio.vad import StreamingVAD


def _feed(vad, amp, ms=400):
    """Feed `ms` of a constant-amplitude tone as one chunk (all windows voiced)."""
    n = int(vad.sr * ms / 1000.0)
    # A tone at the given RMS amplitude; exact shape is irrelevant — only energy.
    chunk = (np.ones(n, dtype="float32") * amp)
    vad.process(chunk)


@pytest.fixture
def voiced_vad(monkeypatch):
    v = StreamingVAD(sample_rate=16000)
    # Force every 32ms window voiced so energy is recorded regardless of Silero.
    monkeypatch.setattr(v, "_probs",
                        lambda audio: [1.0] * (audio.shape[0] // v._WIN))
    return v


def test_falling_tail_is_detected(voiced_vad):
    _feed(voiced_vad, 0.30, ms=400)   # loud body
    _feed(voiced_vad, 0.08, ms=200)   # quiet tail (trailing off)
    assert voiced_vad.tail_falling is True


def test_steady_tail_is_not_falling(voiced_vad):
    _feed(voiced_vad, 0.25, ms=600)   # steady energy throughout
    assert voiced_vad.tail_falling is False


def test_rising_tail_is_not_falling(voiced_vad):
    _feed(voiced_vad, 0.10, ms=400)
    _feed(voiced_vad, 0.30, ms=200)   # louder tail (still going / emphasis)
    assert voiced_vad.tail_falling is False


def test_thin_history_is_neutral(voiced_vad):
    _feed(voiced_vad, 0.20, ms=64)    # only ~2 windows → not enough context
    assert voiced_vad.tail_falling is False


def test_reset_clears_contour(voiced_vad):
    _feed(voiced_vad, 0.30, ms=400)
    _feed(voiced_vad, 0.08, ms=200)
    assert voiced_vad.tail_falling is True
    voiced_vad.reset_utterance()
    assert voiced_vad.tail_falling is False
