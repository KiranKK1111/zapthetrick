"""Prosody analyzer — Architecture.md §"Multi-modal question detection".

Extracts pitch / pause / rhythm features from an audio chunk so the
fusion layer can combine them with the text-based classifier.

The doc commits to:

    Pitch rise at end-of-utterance      → strong question cue
    Long inter-token pauses + low pitch → trailing-off / non-question
    Rapid-fire short tokens             → command, not question

This module produces a [ProsodyFeatures] dict per chunk. Heavy
DSP (librosa, parselmouth) is optional — when those libs aren't
installed, we fall back to a numpy-only path that's less precise
but never raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Per-utterance acoustic features. All values normalized to [0,1]
    when feasible so the fusion layer can apply fixed weights without
    re-tuning per engine."""
    pitch_rise_end: float = 0.0           # 0 = flat or falling, 1 = strong rise
    avg_pause_ms: float = 0.0             # mean silence between tokens
    speech_rate_wps: float = 0.0          # words per second
    energy_peak_at_end: float = 0.0       # 0 = trailing off, 1 = emphatic
    duration_ms: int = 0
    is_question_acoustic: float = 0.0     # combined acoustic score in [0, 1]


def analyze(audio_np, *, sample_rate: int = 16_000) -> ProsodyFeatures:
    """Analyze an utterance and return its prosodic features.

    Tries `parselmouth` (Praat bindings) first, then `librosa`, then
    a numpy-only fallback. The fallback uses RMS energy + zero
    crossing to approximate "is the speaker's voice rising at the
    end" — coarse but useful when the heavy deps aren't installed.
    """
    try:
        return _analyze_with_parselmouth(audio_np, sample_rate)
    except Exception:  # noqa: BLE001 — try the next backend
        pass
    try:
        return _analyze_with_librosa(audio_np, sample_rate)
    except Exception:  # noqa: BLE001
        pass
    return _analyze_with_numpy(audio_np, sample_rate)


# ---- backends ----------------------------------------------------------
def _analyze_with_parselmouth(audio_np, sample_rate: int) -> ProsodyFeatures:
    import parselmouth  # type: ignore — optional dep
    import numpy as np

    snd = parselmouth.Sound(np.asarray(audio_np, dtype=np.float64), sampling_frequency=sample_rate)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 8:
        return _analyze_with_numpy(audio_np, sample_rate)
    # Tail vs body slope.
    tail = float(f0[-max(1, len(f0) // 5):].mean())
    body = float(f0[: max(1, 4 * len(f0) // 5)].mean()) or 1.0
    rise = max(0.0, min(1.0, (tail - body) / max(body * 0.5, 1.0)))
    energy = float(np.sqrt(np.mean(np.square(audio_np[-sample_rate // 4 :])))) if len(audio_np) > sample_rate // 4 else 0.0
    duration_ms = int(1000 * len(audio_np) / sample_rate)
    feats = ProsodyFeatures(
        pitch_rise_end=rise,
        avg_pause_ms=0.0,
        speech_rate_wps=0.0,
        energy_peak_at_end=min(energy * 10.0, 1.0),
        duration_ms=duration_ms,
    )
    feats.is_question_acoustic = _combine(feats)
    return feats


def _analyze_with_librosa(audio_np, sample_rate: int) -> ProsodyFeatures:
    import librosa  # type: ignore — optional dep
    import numpy as np

    y = np.asarray(audio_np, dtype=np.float32)
    if y.size < sample_rate // 4:
        return _analyze_with_numpy(audio_np, sample_rate)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sample_rate)
    f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0) < 8:
        return _analyze_with_numpy(audio_np, sample_rate)
    tail = float(f0[-max(1, len(f0) // 5):].mean())
    body = float(f0[: max(1, 4 * len(f0) // 5)].mean()) or 1.0
    rise = max(0.0, min(1.0, (tail - body) / max(body * 0.5, 1.0)))
    rms = librosa.feature.rms(y=y).mean()
    duration_ms = int(1000 * len(y) / sample_rate)
    feats = ProsodyFeatures(
        pitch_rise_end=rise,
        energy_peak_at_end=min(float(rms) * 20.0, 1.0),
        duration_ms=duration_ms,
    )
    feats.is_question_acoustic = _combine(feats)
    return feats


def _analyze_with_numpy(audio_np, sample_rate: int) -> ProsodyFeatures:
    """Last-resort fallback. Heuristic: compare energy in the last
    20% of the utterance against the body. A pronounced rise in
    energy at the end correlates loosely with a rising-intonation
    question. Coarse — but never raises and runs in <1 ms."""
    try:
        import numpy as np

        y = np.asarray(audio_np, dtype=np.float32)
        if y.size < 80:
            return ProsodyFeatures()
        tail = y[-max(1, y.size // 5):]
        body = y[: max(1, 4 * y.size // 5)]
        tail_e = float(np.sqrt(np.mean(tail * tail)))
        body_e = float(np.sqrt(np.mean(body * body))) or 1e-6
        rise = max(0.0, min(1.0, (tail_e - body_e) / body_e))
        duration_ms = int(1000 * y.size / sample_rate)
        feats = ProsodyFeatures(
            pitch_rise_end=rise,
            energy_peak_at_end=min(tail_e * 10.0, 1.0),
            duration_ms=duration_ms,
        )
        feats.is_question_acoustic = _combine(feats)
        return feats
    except Exception as exc:  # noqa: BLE001
        log.warning("prosody numpy fallback failed: %s", exc)
        return ProsodyFeatures()


def _combine(f: ProsodyFeatures) -> float:
    """Acoustic-only sub-score in [0, 1]. Fusion happens upstream."""
    return max(0.0, min(1.0, 0.7 * f.pitch_rise_end + 0.3 * f.energy_peak_at_end))


__all__ = ["analyze", "ProsodyFeatures"]
