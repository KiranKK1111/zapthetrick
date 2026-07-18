"""
Audio stream processor for the live-listen pipeline.

Receives PCM chunks (from the Flutter client or the server-side capture
module), gates them through Silero VAD, accumulates speech segments, and
emits utterance strings via a callback when a silence gap exceeds
`endpoint_silence_ms`.

This is the bridge between raw audio and the question classifier.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Awaitable, Callable

from app.audio import vad
from app.core.config_loader import cfg
from app.live.hypothesis import completeness
from app.stt import factory as stt_factory


# The segmenter's callback receives the transcribed text plus the
# raw float32 PCM that produced it. Audio is included so the
# question classifier can fuse prosody features (Architecture.md
# §"Multi-modal question detection"). Callers that don't need the
# audio can ignore the second arg.
Callback = Callable[[str, "object"], Awaitable[None]]


@dataclass
class _Buffers:
    """Rolling state for the segmenter."""
    audio: list = field(default_factory=list)         # list of np arrays
    samples: int = 0                                  # total samples in `audio`
    voiced_samples: int = 0                           # samples from VOICED chunks
    last_speech_at: float | None = None
    started_at: float | None = None                   # first voiced chunk time
    last_silence_emit_at: float = 0.0
    last_partial_at: float = 0.0                      # last partial-STT snapshot
    end_partial_voiced: int = 0                       # end-of-speech partial marker


class AudioStreamSegmenter:
    """VAD-gated segmenter that emits transcribed utterances.

    Usage:
        seg = AudioStreamSegmenter(on_utterance=handle_utterance)
        async for chunk in capture.read_chunks():
            await seg.push(chunk)

    `on_utterance` receives the transcribed text whenever the segmenter
    decides an utterance is complete (silence gap or max-utterance cap).

    **Why STT runs off the critical path:** transcription is a network call
    (cloud Whisper, ~0.3-1s). Awaiting it *inside* `push` would block the
    WebSocket receive loop that feeds chunks in, so incoming audio backs up
    and is dropped — the classic "it stops hearing me while it thinks"
    glitch. Instead, `push` snapshots the finished utterance, resets the
    buffer, and spawns a background task to transcribe + emit. `push` then
    returns immediately so the next audio frame is read without delay.
    """

    def __init__(self, on_utterance: Callback, prompt_provider=None,
                 on_partial=None, on_stt_status=None, on_speech_start=None):
        self._on_utterance = on_utterance
        # Optional async callback fired when a NEW utterance begins (first
        # voiced chunk after silence). The live route uses it to pre-warm the
        # LLM provider connection while the speaker is still talking.
        self._on_speech_start = on_speech_start
        # Optional callable returning the current Whisper biasing prompt
        # (technical seed + recent interview questions). Called per utterance
        # so the bias adapts as the conversation builds. May return None.
        self._prompt_provider = prompt_provider
        # Optional async callback for STREAMING partial transcripts: while the
        # speaker is still talking, the growing buffer is transcribed by the
        # FAST provider (cfg.stt.partial_provider) every partial_interval_ms
        # and the interim text is emitted here. The final utterance still goes
        # through the accurate primary chain via `on_utterance`.
        self._on_partial = on_partial
        # Optional async callback (kind: str, detail: str) so a swallowed STT
        # failure/empty result becomes VISIBLE to the client instead of the
        # mic looking dead ("heard you, but couldn't transcribe that").
        self._on_stt_status = on_stt_status
        self._buf = _Buffers()
        self._lock = asyncio.Lock()
        # In-flight transcription tasks (one per finalised utterance). Tracked
        # so `flush` can drain them and teardown can cancel them.
        self._tasks: set[asyncio.Task] = set()
        # At most ONE partial transcription in flight; overlapping partials
        # would just re-transcribe the same prefix and waste CPU.
        self._partial_task: asyncio.Task | None = None
        self._last_partial_text: str = ""
        # (voiced_samples_covered, text) of the newest completed partial for
        # the CURRENT utterance. When no NEW voiced audio arrived after its
        # snapshot and the partial engine is the final engine, the final pass
        # reuses it instead of re-transcribing the same speech. (Keyed on
        # VOICED samples: the buffer also carries silence/pre-roll padding,
        # which adds no words.)
        self._partial_ready: tuple[int, str] | None = None
        # Utterance generation: bumped on every buffer reset so a partial
        # that completes after its utterance finalized is recognized as
        # stale and can't leak text into the next utterance.
        self._utt_gen = 0
        # PRE-ROLL ring: the last ~preroll_ms of pre-speech audio, prepended
        # when an utterance starts so a soft onset the VAD only caught
        # mid-word isn't clipped from the STT input.
        self._preroll: deque = deque()
        self._preroll_samples = 0

    async def _emit_stt_status(self, kind: str, detail: str) -> None:
        """Surface an STT failure/empty to the client (best-effort)."""
        if self._on_stt_status is None:
            return
        try:
            await self._on_stt_status(kind, detail)
        except Exception:  # noqa: BLE001 — status reporting must never break flow
            pass

    def _spawn_transcribe(self, full, ready: tuple[int, str] | None = None,
                          voiced_n: int = -1) -> None:
        """Fire-and-forget transcription of one finalised utterance.

        `ready` is the (voiced_samples, text) of the newest completed
        partial, captured by the caller before the buffer reset; `voiced_n`
        is the finalized utterance's voiced-sample count. When they match
        (no NEW speech after the partial's snapshot — trailing silence adds
        no words) and the partial engine is the final engine, the partial's
        text IS the final transcript — emit it directly and keep the
        redundant re-transcription off the critical path."""
        if ready is not None and self._can_reuse_partial(ready, voiced_n):
            task = asyncio.create_task(self._emit_final(ready[1], full, None))
        else:
            task = asyncio.create_task(self._transcribe_and_emit(full))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _can_reuse_partial(ready: tuple[int, str], voiced_n: int) -> bool:
        """True when a completed partial can stand in for the final pass."""
        if not bool(getattr(cfg.stt, "final_from_partial", True)):
            return False
        # The dual arbitrator produces a confidence score a partial lacks.
        if getattr(cfg.stt, "dual_engine_enabled", False):
            return False
        # A different (faster/weaker) partial engine means the final chain
        # is the accuracy authority — never substitute.
        if (getattr(cfg.stt, "partial_provider", "") or "") != cfg.stt.provider:
            return False
        n_samples, text = ready
        return bool(text) and voiced_n >= 0 and n_samples == voiced_n

    async def _transcribe_and_emit(self, full) -> None:
        """Transcribe one utterance and hand it to the callback. Runs OUTSIDE
        the segmenter lock so it never blocks chunk intake."""
        prompt = None
        if self._prompt_provider is not None:
            try:
                prompt = self._prompt_provider()
            except Exception:  # noqa: BLE001 — biasing is best-effort
                prompt = None
        try:
            text, stt_conf = await stt_factory.transcribe_with_confidence(
                full, prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed STT must not kill the stream
            import logging
            logging.getLogger("zapthetrick.live").exception("STT failed for utterance")
            # Surface the failure so the mic doesn't just silently do nothing.
            await self._emit_stt_status(
                "error", "Couldn't transcribe that — the speech recognizer "
                "failed. If this repeats, check the STT model.")
            return
        if not (text and text.strip()):
            # Heard speech but got NOTHING back (empty/blank). Tell the client
            # so a hot mic that produces no text is explainable, not mysterious.
            await self._emit_stt_status(
                "empty", "Heard you, but couldn't make out any words — "
                "try speaking a little louder or closer to the mic.")
            return
        await self._emit_final(text.strip(), full, stt_conf)

    async def _emit_final(self, text: str, full, stt_conf) -> None:
        """Hand a final transcript to the utterance callback.

        Passes the STT confidence to callbacks that declare `stt_conf`
        (keyword-only, so legacy positional callbacks keep working)."""
        _wants_conf = False
        try:
            import inspect
            params = inspect.signature(self._on_utterance).parameters
            _wants_conf = ("stt_conf" in params or any(
                p.kind == p.VAR_KEYWORD for p in params.values()))
        except Exception:  # noqa: BLE001
            _wants_conf = False
        if _wants_conf:
            await self._on_utterance(text, full, stt_conf=stt_conf)
        else:
            await self._on_utterance(text, full)

    def _maybe_spawn_partial(self, snapshot, voiced_n: int,
                             *, chain: bool = False) -> None:
        """Transcribe the in-progress buffer with the FAST provider and emit
        interim text. One at a time; failures are silent (partials are UX
        sugar — the final pass is authoritative).

        `voiced_n` is the utterance's voiced-sample count at snapshot time —
        the key used to decide whether the final pass can reuse this text.
        `chain=True` (the end-of-speech partial) queues behind an in-flight
        partial instead of being dropped: that last snapshot carries the
        trailing '?' that unlocks the early endpoint gap and speculation,
        and it is what the final pass reuses to skip re-transcription."""
        if self._on_partial is None:
            return
        prev = self._partial_task
        if prev is not None and not prev.done() and not chain:
            return
        gen = self._utt_gen
        n_samples = int(voiced_n)

        async def _run() -> None:
            if chain and prev is not None and not prev.done():
                try:
                    await prev
                except Exception:  # noqa: BLE001
                    pass
            if gen != self._utt_gen:
                return  # utterance already finalized — this snapshot is stale
            try:
                text = await stt_factory.transcribe_partial(snapshot)
            except Exception:  # noqa: BLE001 — never let a partial kill intake
                return
            text = (text or "").strip()
            if not text or gen != self._utt_gen:
                return
            # Remember what this partial covered so a finalize with no newer
            # voiced audio can reuse it as the final transcript.
            self._partial_ready = (n_samples, text)
            if text != self._last_partial_text:
                self._last_partial_text = text
                try:
                    await self._on_partial(text)
                except Exception:  # noqa: BLE001
                    pass

        self._partial_task = asyncio.create_task(_run())

    def _svad(self):
        """Lazy per-segmenter StreamingVAD (stateful Silero, audio-time)."""
        if getattr(self, "_streaming_vad", None) is None:
            self._streaming_vad = vad.StreamingVAD()
        return self._streaming_vad

    def _required_gap_ms(self) -> float:
        """ADAPTIVE end-of-speech gap in AUDIO time.

        * a SHORT utterance so far ("What") is very likely mid-question →
          demand the longer, clearly intentional pause before finalizing;
        * a longer utterance finalizes on the normal gap;
        * a longer utterance whose latest PARTIAL already reads as a complete
          question ("… in Kafka?") finalizes EARLY — the speaker has asked,
          every extra ms of waiting is pure answer latency;
        * a partial that reads MID-THOUGHT ("Can you tell me" — even when the
          ASR guessed a '?') waits LONGER, so a thinking pause doesn't chop
          the question into fragments at the segmenter level.
        """
        base = float(cfg.audio.endpoint_silence_ms)
        svad = self._svad()
        min_speech = float(getattr(cfg.audio, "min_utterance_ms", 700))
        if svad.speech_ms < min_speech:
            return float(getattr(cfg.audio, "short_utterance_gap_ms", 1200))
        txt = (self._last_partial_text or "").rstrip()
        if not txt:
            return base
        kind = completeness(txt)
        if kind == "complete":
            return max(280.0, base * 0.55)
        if kind == "incomplete":
            return float(getattr(cfg.audio, "incomplete_gap_ms", 1200))
        return base

    async def push(self, audio_chunk) -> None:
        """Append a chunk and, when an utterance finalises, spawn its
        transcription. Returns promptly — STT runs in the background.

        Endpointing runs on the stateful StreamingVAD in AUDIO time (32 ms
        window precision, immune to network jitter), with hysteresis so
        intra-word dips don't flicker the speech state."""
        try:
            import numpy as np
        except ImportError:
            return  # numpy isn't installed; nothing to do
        full = None
        final_ready: tuple[int, str] | None = None
        partial_snapshot = None
        end_partial = False
        speech_started = False
        voiced_snapshot = 0
        async with self._lock:
            svad = self._svad()
            has_voice = await asyncio.to_thread(svad.process, audio_chunk)
            now = monotonic()
            n_samples = int(np.asarray(audio_chunk).size)
            if has_voice:
                if not self._buf.audio:
                    self._buf.started_at = now
                    speech_started = True
                    # PRE-ROLL: prepend the recent pre-speech audio so a soft
                    # onset the VAD only caught mid-word isn't clipped.
                    for c in self._preroll:
                        self._buf.audio.append(c)
                        self._buf.samples += int(np.asarray(c).size)
                    self._preroll.clear()
                    self._preroll_samples = 0
                self._buf.audio.append(audio_chunk)
                self._buf.samples += n_samples
                self._buf.voiced_samples += n_samples
                self._buf.last_speech_at = now
                # Force-emit a very long, gap-free utterance so it still gets
                # transcribed (long question, or a room that never goes
                # quiet). The samples guard bounds a pause-heavy buffer too.
                if (svad.speech_ms >= cfg.audio.max_utterance_ms
                        or self._buf.samples >= self._max_buffer_samples()):
                    full = np.concatenate(self._buf.audio).astype(np.float32)
                    voiced_snapshot = self._buf.voiced_samples
                    final_ready = self._reset_utterance_state()
                    svad.reset_utterance()
                elif (
                    self._on_partial is not None
                    and svad.speech_ms >= getattr(cfg.audio, "partial_min_ms", 900)
                    and (now - self._buf.last_partial_at) * 1000.0
                        >= getattr(cfg.audio, "partial_interval_ms", 1200)
                ):
                    # Streaming partials: snapshot the growing utterance for
                    # an interim fast-provider transcription.
                    self._buf.last_partial_at = now
                    voiced_snapshot = self._buf.voiced_samples
                    partial_snapshot = np.concatenate(
                        self._buf.audio).astype(np.float32)
            elif self._buf.audio:
                # Utterance in progress but this chunk scored unvoiced: KEEP
                # it anyway. A softly-spoken word the VAD gated below its
                # threshold must still reach STT — dropping the chunk used to
                # silently delete low-volume words from the transcript.
                self._buf.audio.append(audio_chunk)
                self._buf.samples += n_samples
                # Silence after speech — end-of-speech check in AUDIO time.
                if svad.speech_ended(self._required_gap_ms()):
                    # Snapshot + reset under the lock, transcribe outside it.
                    full = np.concatenate(self._buf.audio).astype(np.float32)
                    voiced_snapshot = self._buf.voiced_samples
                    final_ready = self._reset_utterance_state()
                    svad.reset_utterance()
                elif (
                    self._on_partial is not None
                    and self._buf.voiced_samples > self._buf.end_partial_voiced
                    and svad.trailing_silence_ms >= float(
                        getattr(cfg.audio, "end_partial_trailing_ms", 160))
                ):
                    # END-OF-SPEECH partial: the interval cadence can leave the
                    # newest partial up to partial_interval_ms stale, so the
                    # trailing '?' that unlocks the early endpoint gap (and
                    # speculative answering) is often unseen. At the first sign
                    # of trailing silence, snapshot once — the refreshed text
                    # lands while the endpoint gap is still counting, and the
                    # final pass can reuse it outright.
                    self._buf.end_partial_voiced = self._buf.voiced_samples
                    self._buf.last_partial_at = now
                    voiced_snapshot = self._buf.voiced_samples
                    partial_snapshot = np.concatenate(
                        self._buf.audio).astype(np.float32)
                    end_partial = True
            else:
                # Idle: maintain the pre-roll ring of recent audio.
                self._preroll.append(audio_chunk)
                self._preroll_samples += n_samples
                cap = int(cfg.audio.sample_rate
                          * float(getattr(cfg.audio, "preroll_ms", 240))
                          / 1000.0)
                while self._preroll_samples > cap and len(self._preroll) > 1:
                    dropped = self._preroll.popleft()
                    self._preroll_samples -= int(np.asarray(dropped).size)
        if speech_started and self._on_speech_start is not None:
            self._spawn_speech_start()
        if full is not None:
            self._spawn_transcribe(full, final_ready, voiced_snapshot)
        elif partial_snapshot is not None:
            self._maybe_spawn_partial(partial_snapshot, voiced_snapshot,
                                      chain=end_partial)

    def _max_buffer_samples(self) -> int:
        """Hard cap on one utterance's buffered samples (2x max_utterance —
        the buffer also carries intra-utterance pauses and pre-roll)."""
        return int(cfg.audio.sample_rate
                   * float(cfg.audio.max_utterance_ms) / 1000.0 * 2)

    def utterance_pending(self) -> bool:
        """True while audio is buffered for an utterance that hasn't
        finalized yet. The turn-taking settle timer holds its commit while
        this is true — the speaker resumed, a continuation is coming."""
        return bool(self._buf.audio)

    def _reset_utterance_state(self) -> tuple[int, str] | None:
        """Fresh buffer + invalidate partial state (call under the lock).
        Returns the outgoing utterance's completed-partial record so the
        finalize path can still reuse it as the final transcript."""
        ready = self._partial_ready
        self._buf = _Buffers()
        self._last_partial_text = ""
        self._partial_ready = None
        self._utt_gen += 1
        return ready

    def _spawn_speech_start(self) -> None:
        """Fire-and-forget the speech-start hook (never blocks intake)."""

        async def _run() -> None:
            try:
                await self._on_speech_start()
            except Exception:  # noqa: BLE001 — a warm-up must never break audio
                pass

        task = asyncio.create_task(_run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def flush(self) -> None:
        """Force-emit whatever is in the buffer (e.g. on session end) and wait
        for all in-flight transcriptions to finish."""
        try:
            import numpy as np
        except ImportError:
            return
        full = None
        async with self._lock:
            if self._buf.audio:
                full = np.concatenate(self._buf.audio).astype(np.float32)
                voiced_now = self._buf.voiced_samples
                ready = self._reset_utterance_state()
            if getattr(self, "_streaming_vad", None) is not None:
                self._streaming_vad.reset_utterance()
        if full is not None:
            self._spawn_transcribe(full, ready, voiced_now)
        # Drain any in-flight transcriptions so the caller (session end) sees
        # every utterance emitted before returning.
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def cancel(self) -> None:
        """Drop the buffer and cancel in-flight transcriptions WITHOUT emitting.

        Used on an abrupt disconnect: the socket is already dead, so finishing
        transcription only to fail the send back is wasted work and log noise."""
        async with self._lock:
            self._reset_utterance_state()
            if getattr(self, "_streaming_vad", None) is not None:
                self._streaming_vad.reset_utterance()
        pending = [t for t in self._tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
