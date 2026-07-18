"""Adaptive audio endpointing on the StreamingVAD-based segmenter.

A SHORT utterance + brief gap must NOT be finalized alone — it keeps
accumulating so 'What <breath> is microservices?' is transcribed as ONE
utterance, not cut down to 'What'. A complete question ('… in Kafka?')
finalizes EARLY once the latest partial confirms it.

Drives the real AudioStreamSegmenter with a fake StreamingVAD (audio-time
semantics, driven by the chunk's first sample: 1.0=voice, 0.0=silence) and
mocked STT — deterministic, no torch/model needed."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.audio import stream as stream_mod
from app.audio.stream import AudioStreamSegmenter

SR = 16000


class FakeStreamingVAD:
    """Audio-time VAD double: each chunk counts its samples as voice/silence
    based on the first sample value (1.0 voiced / 0.0 silence)."""

    def __init__(self, *a, **k):
        self.sr = SR
        self.speaking = False
        self._speech = 0
        self._silence = 0

    def process(self, chunk) -> bool:
        n = np.asarray(chunk).reshape(-1).shape[0]
        voiced = bool(np.asarray(chunk).reshape(-1)[0] > 0.5)
        if voiced:
            self.speaking = True
            self._speech += n
            self._silence = 0
        else:
            self.speaking = False
            self._silence += n
        return voiced

    @property
    def speech_ms(self):
        return self._speech * 1000.0 / self.sr

    @property
    def trailing_silence_ms(self):
        return self._silence * 1000.0 / self.sr

    def speech_ended(self, min_gap_ms):
        return self._speech > 0 and self.trailing_silence_ms >= min_gap_ms

    def reset_utterance(self):
        self._speech = 0
        self._silence = 0
        self.speaking = False


def _drive(monkeypatch, *, script, transcript="What is microservices?",
           partial_text=None):
    """`script` = list of (is_voice, ms) audio segments. Returns utterances."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    async def fake_transcribe(audio, prompt=None):
        return transcript, None
    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_transcribe)

    got: list[str] = []

    async def on_utt(text, audio, **kw):
        got.append(text)

    seg = AudioStreamSegmenter(on_utterance=on_utt)
    if partial_text is not None:
        seg._last_partial_text = partial_text

    async def go():
        for is_voice, ms in script:
            n = int(SR * ms / 1000)
            chunk = np.full(n, 1.0 if is_voice else 0.0, dtype=np.float32)
            await seg.push(chunk)
        # do NOT flush for split-tests — flush force-emits the buffer.
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())
    return got, seg


def test_short_utterance_brief_gap_is_not_split(monkeypatch):
    # "What" (0.3s voice), a 0.6s breath (< short_utterance_gap 1.2s): the
    # short utterance must NOT finalize; speech continues and only the real
    # 1.4s end pause finalizes ONE utterance.
    script = [
        (True, 300),                 # "What"
        (False, 600),                # breath — must NOT finalize (short)
        (True, 900),                 # "is microservices?"
        (False, 1400),               # real end pause → finalize once
    ]
    got, _ = _drive(monkeypatch, script=script)
    assert len(got) == 1, got


def test_long_utterance_finalizes_on_normal_gap(monkeypatch):
    script = [
        (True, 1200),                # 1.2s speech (>= min_utterance 700ms)
        (False, 700),                # 700ms ≥ endpoint 650ms → finalize
    ]
    got, _ = _drive(monkeypatch, script=script)
    assert len(got) == 1, got


def test_short_utterance_with_intentional_long_pause_finalizes(monkeypatch):
    script = [
        (True, 400),                 # short speech
        (False, 1400),               # 1.4s pause ≥ short gap 1.2s → finalize
    ]
    got, _ = _drive(monkeypatch, script=script)
    assert len(got) == 1, got


def test_complete_question_partial_finalizes_early(monkeypatch):
    # When the latest PARTIAL already ends with '?', the gap needed shrinks
    # (max(280, 650*0.55)=357ms) — the answer starts sooner.
    script = [
        (True, 1200),
        (False, 400),                # 400ms ≥ 357ms early gap → finalize
    ]
    got, _ = _drive(monkeypatch, script=script,
                    partial_text="how would you scale Kafka?")
    assert len(got) == 1, got


def test_incomplete_partial_does_not_finalize_early(monkeypatch):
    # Same 400ms gap but the partial does NOT read complete → normal 650ms
    # gap applies → nothing finalizes yet.
    script = [
        (True, 1200),
        (False, 400),
    ]
    got, _ = _drive(monkeypatch, script=script,
                    partial_text="how would you scale")
    assert len(got) == 0, got
