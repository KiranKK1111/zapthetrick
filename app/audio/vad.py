"""
Silero Voice Activity Detection.

Distinguishes speech from non-speech (air conditioner, keyboard clicks,
silence). 1MB model, runs in real time on CPU. We use it as a gate before
STT so we don't waste cycles transcribing silence or background noise.

The model expects 16 kHz mono float32 samples; the audio pipeline
normalises input before calling here.

**Degradation:** if Silero / torch is unavailable (e.g. a slim/bundled deploy
that didn't ship torch), we DO NOT crash the live audio pipeline — we fall
back to a dependency-free energy (RMS) gate so transcription still works.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config_loader import cfg


log = logging.getLogger("zapthetrick.vad")


class VADError(RuntimeError):
    pass


# Flips to True the first time Silero fails to load/run, so we stop retrying
# the heavy path and use the energy gate for the rest of the session.
_silero_failed = False


@lru_cache(maxsize=1)
def _model():
    """Load Silero VAD lazily. The first call downloads ~1MB of weights."""
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise VADError(
            "silero-vad is not installed. Run: pip install silero-vad torch"
        ) from exc
    return load_silero_vad()


def detect_speech(audio_np) -> list[tuple[float, float]]:
    """Return (start_s, end_s) intervals of speech in the audio array."""
    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise VADError(
            "silero-vad is not installed. Run: pip install silero-vad torch"
        ) from exc
    import torch

    audio_tensor = torch.from_numpy(audio_np).float()
    stamps = get_speech_timestamps(
        audio_tensor,
        _model(),
        threshold=cfg.audio.vad_threshold,
        sampling_rate=cfg.audio.sample_rate,
        return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def _energy_has_speech(audio_np) -> bool:
    """Dependency-free fallback gate: RMS energy over a threshold = speech.

    Crude vs Silero (it can't tell speech from loud noise) but it needs no
    torch and keeps the live pipeline working on a slim deploy. The
    downstream question-prediction agent filters non-questions anyway."""
    import numpy as np

    a = np.ascontiguousarray(np.asarray(audio_np, dtype="float32").reshape(-1))
    if a.size == 0:
        return False
    rms = float(np.sqrt(np.mean(a * a)))
    # ~ -40 dBFS. Override via cfg.audio.energy_threshold if present.
    floor = float(getattr(cfg.audio, "energy_threshold", 0.0) or 0.0) or 0.01
    return rms >= floor


def _silero_has_speech(audio_np) -> bool:
    import numpy as np
    import torch

    sr = cfg.audio.sample_rate
    win = 512 if sr >= 16_000 else 256  # Silero's required window sizes
    audio = np.ascontiguousarray(np.asarray(audio_np, dtype="float32").reshape(-1))
    if audio.shape[0] == 0:
        return False
    if audio.shape[0] < win:
        audio = np.pad(audio, (0, win - audio.shape[0]))

    model = _model()
    threshold = cfg.audio.vad_threshold
    try:
        model.reset_states()
    except Exception:  # noqa: BLE001 -- not all builds expose it
        pass

    n_windows = audio.shape[0] // win
    for i in range(n_windows):
        window = torch.from_numpy(audio[i * win : (i + 1) * win])
        prob = float(model(window, sr).item())
        if prob >= threshold:
            return True
    return False


def has_speech(audio_np) -> bool:
    """Cheap predicate: 'does this chunk contain any speech at all?'

    Tries Silero's raw per-window speech probability (accurate); on ANY
    failure (torch/silero missing, model load error, inference error) it
    permanently falls back to the energy gate so a slim/broken deploy can't
    crash the live audio socket — it just transcribes a bit less selectively.
    """
    global _silero_failed
    if not _silero_failed:
        try:
            return _silero_has_speech(audio_np)
        except Exception as exc:  # noqa: BLE001
            _silero_failed = True
            log.warning(
                "Silero VAD unavailable (%s) — falling back to energy-based "
                "VAD for the rest of this run.", exc,
            )
    return _energy_has_speech(audio_np)


class StreamingVAD:
    """STATEFUL streaming VAD — one instance per live session.

    The old per-chunk `has_speech` was a weak use of a strong model: it
    RESET Silero's recurrent state on every chunk (losing the acoustic
    context that makes Silero accurate on continuations), collapsed a whole
    chunk to one binary bit, and left endpointing to wall-clock arrival
    times. This class runs Silero the way it is designed to be run:

      * the recurrent state persists ACROSS chunks (true streaming);
      * every 32 ms window gets a probability, with HYSTERESIS — speech
        starts at `start_threshold` but only ends below the lower
        `end_threshold`, so natural intra-word dips don't flicker;
      * silence is integrated in AUDIO TIME (samples), immune to network
        jitter/bursty chunk arrival;
      * it answers the three endpointing questions directly:
          is someone speaking?   → `speaking`
          is there silence?      → `trailing_silence_ms`
          has the speaker stopped? → `speech_ended(min_gap_ms)`

    Fail-open: any Silero failure downgrades to the RMS energy gate per
    window, so the live socket never dies with the VAD.
    """

    _WIN = 512  # Silero's native window at 16 kHz = 32 ms

    def __init__(self, sample_rate: int | None = None,
                 start_threshold: float | None = None,
                 end_threshold: float | None = None):
        self.sr = int(sample_rate or cfg.audio.sample_rate)
        base = float(start_threshold if start_threshold is not None
                     else cfg.audio.vad_threshold)
        self.start_threshold = base
        # Hysteresis: keep "speaking" until the probability drops well below
        # the start gate (Silero-recommended pattern: end ≈ start - 0.15).
        self.end_threshold = float(
            end_threshold if end_threshold is not None
            else max(0.15, base - 0.15))
        self.speaking = False
        self._trailing_silence_samples = 0
        self._speech_samples = 0
        self._silero_ok = True
        self._model = None
        self._pending = None  # leftover < one window, carried to next chunk
        # Samples since the last voiced window ACROSS utterances (None until
        # the first voice). Drives the re-entry threshold: speakers routinely
        # drop volume on a continuation after a pause ("…tell me <pause> in
        # spring boot"), and a fixed start gate silently discards those soft
        # words. Within `vad_reentry_window_ms` of prior speech the start
        # threshold relaxes toward the end threshold.
        self._samples_since_voice: int | None = None
        self._reentry_window_ms = float(
            getattr(cfg.audio, "vad_reentry_window_ms", 3000) or 0)
        self._reentry_delta = float(
            getattr(cfg.audio, "vad_reentry_delta", 0.15) or 0.0)

    # -- internals ----------------------------------------------------------
    def _probs(self, audio):
        """Per-window speech probabilities for `audio` (float32 mono)."""
        import numpy as np
        if self._silero_ok and not _silero_failed:
            try:
                import torch
                if self._model is None:
                    # A PRIVATE model instance per session: Silero's recurrent
                    # state is what makes streaming accurate, and sharing the
                    # module singleton would interleave state across concurrent
                    # live sessions. The model is ~1 MB — cheap per session.
                    from silero_vad import load_silero_vad
                    self._model = load_silero_vad()
                out = []
                n = audio.shape[0] // self._WIN
                for i in range(n):
                    win = torch.from_numpy(
                        audio[i * self._WIN:(i + 1) * self._WIN])
                    out.append(float(self._model(win, self.sr).item()))
                return out
            except Exception as exc:  # noqa: BLE001
                self._silero_ok = False
                log.warning("StreamingVAD: silero failed (%s) — energy "
                            "fallback for this session.", exc)
        # Energy fallback, per window.
        floor = float(getattr(cfg.audio, "energy_threshold", 0.0) or 0.0) or 0.01
        out = []
        n = audio.shape[0] // self._WIN
        for i in range(n):
            win = audio[i * self._WIN:(i + 1) * self._WIN]
            rms = float(np.sqrt(np.mean(win * win))) if win.size else 0.0
            out.append(1.0 if rms >= floor else 0.0)
        return out

    # -- public API ---------------------------------------------------------
    def process(self, audio_np) -> bool:
        """Feed one chunk; returns True if any window in it was voiced.
        Updates `speaking`, `trailing_silence_ms` and `speech_ms` with
        window (32 ms) precision in audio time."""
        import numpy as np
        audio = np.ascontiguousarray(
            np.asarray(audio_np, dtype="float32").reshape(-1))
        if self._pending is not None and self._pending.size:
            audio = np.concatenate([self._pending, audio])
            self._pending = None
        rem = audio.shape[0] % self._WIN
        if rem:
            self._pending = audio[-rem:]
            audio = audio[:-rem]
        if audio.shape[0] == 0:
            return self.speaking
        voiced_any = False
        for p in self._probs(audio):
            if self.speaking:
                threshold = self.end_threshold
            else:
                threshold = self.start_threshold
                # Re-entry: shortly after prior speech, open the gate at a
                # lower threshold so a quieter continuation is not discarded.
                if (self._samples_since_voice is not None
                        and self._reentry_delta > 0
                        and (self._samples_since_voice * 1000.0 / self.sr)
                        <= self._reentry_window_ms):
                    threshold = max(self.end_threshold,
                                    self.start_threshold - self._reentry_delta)
            if p >= threshold:
                voiced_any = True
                self.speaking = True
                self._speech_samples += self._WIN
                self._trailing_silence_samples = 0
                self._samples_since_voice = 0
            else:
                self._trailing_silence_samples += self._WIN
                if self._samples_since_voice is not None:
                    self._samples_since_voice += self._WIN
                if self.speaking and p < self.end_threshold:
                    self.speaking = False
        return voiced_any

    @property
    def trailing_silence_ms(self) -> float:
        """Audio-time silence since the last voiced window."""
        return self._trailing_silence_samples * 1000.0 / self.sr

    @property
    def speech_ms(self) -> float:
        """Total voiced audio accumulated since the last reset."""
        return self._speech_samples * 1000.0 / self.sr

    def speech_ended(self, min_gap_ms: float) -> bool:
        """Has the speaker stopped for at least `min_gap_ms` of AUDIO time?"""
        return (self._speech_samples > 0
                and self._trailing_silence_samples * 1000.0 / self.sr
                >= min_gap_ms)

    def reset_utterance(self) -> None:
        """Start tracking a fresh utterance (keeps the acoustic model state —
        Silero context carries across utterances by design)."""
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self.speaking = False

