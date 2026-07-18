"""Intelligent utterance merging + vocal-modulation capture.

Pins the fixes for the reported "Can you tell me?" bug — a question spoken
with thinking pauses ("can you… ahh… tell me <2s> various stereotype
annotations <2s> in spring boot") being answered as a fragment:

1. `completeness()` sees through ASR-guessed punctuation: Parakeet appends
   '?' to a dangling stem, which used to make every premature endpoint look
   like a finished question (instant settle, early finalize, speculation).
2. `merged()` stitches fragments into ONE clean sentence.
3. The retroactive continuation detector recognizes late-arriving tails.
4. Hesitation fillers are stripped from transcripts.
5. Low-volume capture: chunks the VAD scored unvoiced are still buffered
   mid-utterance, a pre-roll protects soft onsets, and the VAD re-entry
   threshold reopens the gate for quieter continuations.
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.audio import stream as stream_mod
from app.audio import vad as vad_mod
from app.audio.stream import AudioStreamSegmenter
from app.live.hypothesis import HypothesisBuffer, completeness
from app.live.repair import strip_fillers

SR = 16000


# ── 1. completeness() vs ASR-guessed '?' ────────────────────────────────────

def test_dangling_stem_with_fake_question_mark_is_incomplete():
    # The reported bug: the ASR guesses '?' on a mid-thought stem.
    assert completeness("Can you tell me?") == "incomplete"
    assert completeness("Can you?") == "incomplete"
    assert completeness("What is?") == "incomplete"
    assert completeness("How would you?") == "incomplete"


def test_real_questions_still_read_complete():
    assert completeness("How would you scale Kafka?") == "complete"
    assert completeness("Why should we hire you?") == "complete"
    assert completeness("What motivates you?") == "complete"
    assert completeness("How are you?") == "complete"


def test_settle_no_longer_collapses_on_fake_question_mark():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("Can you tell me?", now=0.0, has_audio=True)
    assert buf.required_settle_ms() == 2100  # incomplete → long merge window


def test_multi_fragment_neutral_keeps_wide_merge_window():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("Can you tell me?", now=0.0, has_audio=True)
    buf.add("various stereotype annotations", now=2.5, has_audio=True)
    # Merged tail is neutral, but the speaker is drip-feeding → 1200ms.
    assert buf.required_settle_ms() == 1200


# ── 2. merged() stitches fragments into one sentence ────────────────────────

def test_merged_drops_artifact_punctuation_between_fragments():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("Can you tell me?", now=0.0, has_audio=True)
    buf.add("various stereotype annotations", now=2.5, has_audio=True)
    buf.add("in spring boot?", now=5.0, has_audio=True)
    assert buf.merged() == (
        "Can you tell me various stereotype annotations in spring boot?")


def test_merged_only_trusts_final_fragment_punctuation():
    # The head's fake '?' must NOT be re-appended: the merged tail would
    # read "complete" and commit before the real ending arrives.
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("Can you tell me?", now=0.0, has_audio=True)
    buf.add("various stereotype annotations", now=2.5, has_audio=True)
    assert buf.merged() == "Can you tell me various stereotype annotations"


def test_merged_dedupes_boundary_word():
    buf = HypothesisBuffer(settle_ms=600)
    buf.add("can you tell me?", now=0.0)
    buf.add("me about kafka?", now=1.0)
    assert buf.merged() == "can you tell me about kafka?"


# ── 3. Retroactive continuation detection + merge ────────────────────────────

def test_looks_like_continuation():
    from app.api.routes_ws import _looks_like_continuation
    assert _looks_like_continuation("in spring boot.")
    assert _looks_like_continuation("In spring boot?")
    assert _looks_like_continuation("and for large payloads?")
    assert _looks_like_continuation("various stereotype annotations")
    # Real new questions / closed statements are NOT continuations.
    assert not _looks_like_continuation("How does Kafka scale?")
    assert not _looks_like_continuation("What about consistency?")
    assert not _looks_like_continuation("Yeah that sounds right.")
    assert not _looks_like_continuation("")


def test_merge_continuation_produces_one_clean_question():
    from app.api.routes_ws import _merge_continuation
    assert _merge_continuation(
        "Can you tell me?", "various stereotype annotations",
    ) == "Can you tell me various stereotype annotations"
    assert _merge_continuation(
        "Can you tell me various stereotype annotations?", "In spring boot.",
    ) == "Can you tell me various stereotype annotations In spring boot."
    # Boundary word deduped; the tail's own '?' survives.
    assert _merge_continuation(
        "can you tell me?", "me about kafka?",
    ) == "can you tell me about kafka?"


def test_speculation_never_fires_on_dangling_stem_or_continuation():
    from app.api.routes_ws import _speculation_worthy
    assert not _speculation_worthy("Can you tell me?")
    assert not _speculation_worthy("In spring boot?")
    assert _speculation_worthy("How would you scale Kafka?")


# ── 4. Filler stripping ──────────────────────────────────────────────────────

def test_strip_fillers():
    assert strip_fillers("can you ahh tell me um about kafka") == (
        "can you tell me about kafka")
    assert strip_fillers("Uh, can you tell me, umm, about Kafka?") == (
        "can you tell me, about Kafka?")
    # Real words containing filler substrings are never touched.
    assert strip_fillers("the umbrella term ahead of us") == (
        "the umbrella term ahead of us")
    # Never empties the text.
    assert strip_fillers("umm") == "umm"


# ── 5. Low-volume capture ─────────────────────────────────────────────────────

class FakeStreamingVAD:
    """Audio-time VAD double (first-sample voiced/unvoiced convention)."""

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


def test_soft_midword_chunks_and_preroll_reach_stt(monkeypatch):
    """Chunks the VAD gated as unvoiced mid-utterance (soft words) must still
    reach STT, and the pre-roll must protect the onset — the transcribed
    audio covers everything, not just the loud chunks."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    sizes: list[int] = []

    async def fake_final(audio, prompt=None):
        sizes.append(int(np.asarray(audio).size))
        return "text", None

    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_final)

    async def on_utt(text, audio, **kw):
        pass

    seg = AudioStreamSegmenter(on_utterance=on_utt)

    async def go():
        await seg.push(_chunk(False, 200))   # idle → pre-roll ring
        await seg.push(_chunk(True, 500))    # speech starts (pre-roll joins)
        await seg.push(_chunk(False, 200))   # soft word scored unvoiced — KEPT
        await seg.push(_chunk(True, 500))    # speech continues
        await seg.push(_chunk(False, 1400))  # end pause → finalize
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())
    total_ms = 200 + 500 + 200 + 500 + 1400
    assert sizes == [int(SR * total_ms / 1000)]


def test_utterance_pending_tracks_buffered_speech(monkeypatch):
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)

    async def fake_final(audio, prompt=None):
        return "text", None

    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        fake_final)

    async def on_utt(text, audio, **kw):
        pass

    seg = AudioStreamSegmenter(on_utterance=on_utt)

    async def go():
        assert not seg.utterance_pending()
        await seg.push(_chunk(True, 800))
        assert seg.utterance_pending()       # speech buffered, not finalized
        await seg.push(_chunk(False, 1400))  # finalize
        assert not seg.utterance_pending()
        pending = [t for t in seg._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(go())


def test_vad_reentry_lowers_start_threshold_for_soft_continuations():
    """After prior speech, a quieter continuation (prob between the relaxed
    and normal start thresholds) must still open the gate — but only within
    the re-entry window."""
    svad = vad_mod.StreamingVAD(sample_rate=SR, start_threshold=0.5)
    seq: list[float] = []
    svad._probs = lambda audio: [seq.pop(0)]

    def feed(p: float) -> bool:
        seq.append(p)
        return svad.process(np.zeros(512, dtype=np.float32))

    assert feed(0.6)            # loud speech opens the gate
    assert not feed(0.2)        # hard silence closes it
    # Soft continuation shortly after speech: 0.4 < 0.5 start threshold but
    # ≥ the relaxed re-entry threshold (0.35) → captured.
    assert feed(0.40)
    assert not feed(0.2)
    # Burn through the 3s re-entry window with silence…
    for _ in range(100):        # 100 × 32ms ≈ 3.2s
        feed(0.1)
    # …then the same soft audio no longer passes: full threshold applies.
    assert not feed(0.40)
