"""Latency fast-path optimizations (stop-speaking → first answer text).

Pins the four code-level levers that collapse the answer's critical path:

1. Settle double-wait removed: an AUDIO-derived hypothesis already sits
   behind the segmenter's VAD endpoint (real silence confirmed in audio
   time), so a complete tail answers immediately and a neutral tail keeps
   only a short merge window. Text fragments keep the full window.
2. End-of-speech partial: at the first sign of trailing silence the
   segmenter snapshots one extra partial (bypassing the interval cadence),
   so the trailing '?' that unlocks the early endpoint gap is seen fresh.
3. Final-from-partial: a completed partial that covered exactly the
   finalized audio (same engine) IS the final transcript — the redundant
   re-transcription is skipped.
4. Speculation trigger widened: imperative questions without a terminal
   '?' ("Tell me about …") now start speculative answers too.

Plus the speech-start LLM connection pre-warm targeting the pinned
live-model's provider.
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.audio import stream as stream_mod
from app.audio.stream import AudioStreamSegmenter
from app.live.hypothesis import HypothesisBuffer

SR = 16000


# ── 1. Settle window collapses behind a VAD-confirmed endpoint ──────────────

def test_settle_collapses_for_vad_confirmed_complete():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("How would you scale Kafka?", now=0.0, has_audio=True)
    assert buf.required_settle_ms() == 0
    assert buf.settle_due(now=0.0)


def test_settle_short_for_vad_confirmed_neutral():
    buf = HypothesisBuffer(settle_ms=600)
    # No terminal punctuation, tail is not grammatically dangling → neutral.
    buf.add("Tell me about your Kafka experience", now=0.0, has_audio=True)
    assert buf.required_settle_ms() == 250


def test_settle_unchanged_for_text_fragments():
    # Text-injected fragments have no VAD behind them → full dynamic waits.
    com = HypothesisBuffer(settle_ms=600)
    com.add("How would you scale Kafka?", now=0.0)
    assert com.required_settle_ms() == 360
    neu = HypothesisBuffer(settle_ms=600)
    neu.add("Tell me about your Kafka experience", now=0.0)
    assert neu.required_settle_ms() == 600


def test_incomplete_still_waits_long_even_with_audio():
    # VAD silence can't tell us a dangling thought is finished — the long
    # window that fixed "What is <pause> Kafka?" must survive.
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("What is", now=0.0, has_audio=True)
    assert buf.required_settle_ms() == 2100


# ── 2+3. End-of-speech partial + final reuses full-coverage partial ─────────

class FakeStreamingVAD:
    """Audio-time VAD double (same contract as test_adaptive_endpointing)."""

    def __init__(self, *a, **k):
        self.sr = SR
        self.speaking = False
        self._speech = 0
        self._silence = 0

    def process(self, chunk) -> bool:
        arr = np.asarray(chunk).reshape(-1)
        voiced = bool(arr[0] > 0.5)
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


def _chunk(is_voice: bool, ms: int):
    n = int(SR * ms / 1000)
    return np.full(n, 1.0 if is_voice else 0.0, dtype=np.float32)


def test_end_partial_fires_and_final_reuses_it(monkeypatch):
    """1.2s speech + 200ms silence → the END-OF-SPEECH partial fires (before
    any interval partial would); its '?' unlocks the early 357ms gap; and the
    finalize reuses the partial text — the final STT pass never runs."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)
    # Reuse requires partial_provider == provider — pin BOTH so the test
    # doesn't depend on whatever STT engine the user last selected in the
    # Settings dropdown (config.yaml is live state, not a fixture).
    from app.core.config_loader import cfg as _cfg
    monkeypatch.setattr(_cfg.stt, "provider", "parakeet", raising=False)
    monkeypatch.setattr(_cfg.stt, "partial_provider", "parakeet",
                        raising=False)
    monkeypatch.setattr(_cfg.stt, "dual_engine_enabled", False,
                        raising=False)

    final_calls: list[int] = []

    async def fake_final(audio, prompt=None):
        final_calls.append(1)
        return "FINAL PASS TEXT", None

    async def fake_partial(audio):
        return "How would you scale Kafka?"

    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_final)
    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_partial",
                        fake_partial)

    got: list[str] = []
    partials: list[str] = []

    async def on_utt(text, audio, **kw):
        got.append(text)

    async def on_partial(text):
        partials.append(text)

    seg = AudioStreamSegmenter(on_utterance=on_utt, on_partial=on_partial)

    async def go():
        await seg.push(_chunk(True, 1200))
        # 200ms trailing silence ≥ end_partial_trailing_ms (160) → end-partial.
        await seg.push(_chunk(False, 200))
        assert seg._partial_task is not None
        await seg._partial_task
        # Partial ends '?' → early gap 357ms; total trailing 400ms → finalize.
        await seg.push(_chunk(False, 200))
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())
    assert partials == ["How would you scale Kafka?"]
    assert got == ["How would you scale Kafka?"]
    assert final_calls == []  # reused the partial — no redundant final pass


def test_final_runs_when_speech_continues_after_partial(monkeypatch):
    """Voiced audio AFTER the last partial → coverage mismatch → the real
    final pass must run (the partial is missing the newest words)."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    async def fake_final(audio, prompt=None):
        return "FINAL PASS TEXT", None

    async def fake_partial(audio):
        return "How would you"

    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_final)
    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_partial",
                        fake_partial)

    got: list[str] = []

    async def on_utt(text, audio, **kw):
        got.append(text)

    async def on_partial(text):
        pass

    seg = AudioStreamSegmenter(on_utterance=on_utt, on_partial=on_partial)

    async def go():
        await seg.push(_chunk(True, 800))
        await seg.push(_chunk(False, 200))       # end-partial on this prefix
        assert seg._partial_task is not None
        await seg._partial_task
        await seg.push(_chunk(True, 600))        # speaker continues
        # Partial reads INCOMPLETE ("How would you") → extended 1200ms gap.
        await seg.push(_chunk(False, 1300))      # ≥1200ms → finalize
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())
    assert got == ["FINAL PASS TEXT"]


def test_speech_start_hook_fires_once_per_utterance(monkeypatch):
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    async def fake_final(audio, prompt=None):
        return "text", None

    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_final)

    starts: list[int] = []

    async def on_utt(text, audio, **kw):
        pass

    async def on_start():
        starts.append(1)

    seg = AudioStreamSegmenter(on_utterance=on_utt, on_speech_start=on_start)

    async def go():
        await seg.push(_chunk(True, 400))
        await seg.push(_chunk(True, 400))   # same utterance — no second fire
        await seg.push(_chunk(False, 1400))  # finalize
        await seg.push(_chunk(True, 300))   # NEW utterance — fires again
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())
    assert starts == [1, 1]


# ── 4. Speculation trigger widened beyond '?' ────────────────────────────────

def test_speculation_worthy():
    from app.api.routes_ws import _speculation_worthy
    # The strong signal: '?' + question heuristic.
    assert _speculation_worthy("How would you scale Kafka?")
    # Imperative questions without a '?' now speculate too.
    assert _speculation_worthy("Tell me about your experience with Kafka.")
    assert _speculation_worthy("Tell me about your experience with Kafka")
    # Dangling tails must NOT speculate (mid-sentence partials).
    assert not _speculation_worthy("How would you")
    assert not _speculation_worthy("Can you explain")
    # Statements must NOT speculate.
    assert not _speculation_worthy("We use Kafka heavily in production")
    assert not _speculation_worthy("")


# ── 5. Speech-start pre-warm targets the pinned live provider ────────────────

def test_warm_live_provider_targets_pinned_provider(monkeypatch):
    import app.core.http_pool as pool
    from app.core.config_loader import cfg
    from app.perceived import prefetch as pf

    calls: list[str] = []

    class FakeClient:
        async def get(self, url, timeout=None):
            calls.append(url)

    monkeypatch.setattr(pool, "get_http_client", lambda: FakeClient())
    monkeypatch.setattr(cfg.llm, "live_model", "llama-3.3-70b-versatile",
                        raising=False)
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", True,
                        raising=False)
    asyncio.run(pf.warm_live_provider())
    assert calls, "warm should ping the provider"
    assert "api.groq.com" in calls[0]
