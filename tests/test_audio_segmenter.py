"""Live-audio segmenter: VAD-gated buffering, endpoint/length emit, and the
NON-BLOCKING transcription contract.

The segmenter must NOT await the (network) STT call inside `push` — that would
stall the WebSocket receive loop and drop incoming audio. These tests pin:
  - a silence gap finalises an utterance,
  - a long gap-free utterance is force-emitted at the max-utterance cap,
  - `push` returns BEFORE transcription completes (STT runs in the background),
  - `flush` drains in-flight transcriptions.

Endpointing runs on the StreamingVAD in AUDIO time (samples), so tests carry
duration in chunk LENGTH: a chunk of N samples is N/16000 seconds of voice
(first sample 1.0) or silence (first sample 0.0).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.audio import stream as stream_mod
from app.audio.stream import AudioStreamSegmenter

SR = 16000


class FakeStreamingVAD:
    """Audio-time VAD double: counts each chunk's samples as voice/silence
    based on its first sample (1.0 voiced / 0.0 silence)."""

    def __init__(self, *a, **k):
        self.sr = SR
        self.speaking = False
        self._speech = 0
        self._silence = 0

    def process(self, chunk) -> bool:
        arr = np.asarray(chunk).reshape(-1)
        voiced = bool(arr.shape[0] and arr[0] > 0.5)
        if voiced:
            self.speaking = True
            self._speech += arr.shape[0]
            self._silence = 0
        else:
            self.speaking = False
            self._silence += arr.shape[0]
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


def _voice(ms):
    return np.ones(int(SR * ms / 1000), dtype=np.float32)


def _silence(ms):
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def _wire(monkeypatch, *, stt_delay=0.0):
    """Patch the StreamingVAD + STT. Returns an event set when STT starts."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    started = asyncio.Event()

    async def _transcribe(audio_np, prompt=None):
        started.set()
        if stt_delay:
            await asyncio.sleep(stt_delay)
        return f"text:{len(audio_np)}", None

    monkeypatch.setattr(
        stream_mod.stt_factory, "transcribe_with_confidence", _transcribe)
    return started


def test_silence_gap_finalises_utterance(monkeypatch):
    emitted: list[tuple[str, int]] = []

    async def on_utt(text, audio):
        emitted.append((text, len(audio)))

    _wire(monkeypatch)

    async def run():
        seg = AudioStreamSegmenter(on_utterance=on_utt)
        await seg.push(_voice(400))     # 0.4s speech (short utterance)
        await seg.push(_voice(400))     # 0.8s total (>= min_utterance 700ms)
        await seg.push(_silence(700))   # 700ms ≥ endpoint 650ms → finalises
        await seg.flush()               # drain background STT

    asyncio.run(run())
    assert len(emitted) == 1
    # The WHOLE utterance reaches STT — voiced chunks AND the mid/trailing
    # audio the VAD scored unvoiced. Dropping unvoiced chunks used to
    # silently delete softly-spoken words from the transcript; silence is
    # harmless to the recognizer.
    n = int(SR * (0.4 + 0.4 + 0.7))
    assert emitted[0] == (f"text:{n}", n)


def test_long_gapless_utterance_force_emitted(monkeypatch):
    emitted: list[str] = []

    async def on_utt(text, audio):
        emitted.append(text)

    _wire(monkeypatch)

    async def run():
        seg = AudioStreamSegmenter(on_utterance=on_utt)
        await seg.push(_voice(10_000))  # 10s continuous speech
        await seg.push(_voice(10_000))  # 20s > 15s cap → force-emit
        await seg.flush()

    asyncio.run(run())
    n = int(SR * 20.0)
    assert emitted == [f"text:{n}"]


def test_push_does_not_block_on_transcription(monkeypatch):
    """`push` must return before STT finishes — STT runs in the background."""
    order: list[str] = []

    async def on_utt(text, audio):
        order.append("emit")

    _wire(monkeypatch, stt_delay=0.2)   # slow transcription

    async def run():
        seg = AudioStreamSegmenter(on_utterance=on_utt)
        await seg.push(_voice(800))     # voiced (>= min_utterance)
        await seg.push(_silence(700))   # endpoint → spawns background STT
        order.append("push_returned")
        # STT may have started but cannot have emitted yet (still sleeping).
        assert "emit" not in order
        await seg.flush()               # waits for the background STT
        assert "emit" in order

    asyncio.run(run())
    # push returned before the emit happened.
    assert order.index("push_returned") < order.index("emit")


def test_cancel_drops_buffer_without_emitting(monkeypatch):
    """On abrupt disconnect, `cancel` must not emit a final transcript."""
    emitted: list[str] = []

    async def on_utt(text, audio):
        emitted.append(text)

    _wire(monkeypatch)

    async def run():
        seg = AudioStreamSegmenter(on_utterance=on_utt)
        await seg.push(_voice(400))
        await seg.push(_voice(100))     # still mid-utterance (no endpoint)
        await seg.cancel()              # socket died — drop everything
        await seg.flush()               # nothing left to emit
        return seg

    asyncio.run(run())
    assert emitted == []
