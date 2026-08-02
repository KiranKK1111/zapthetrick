"""Echo canceller — MEASURED, not asserted (design §Noise).

An AEC that "runs" but cancels nothing is worse than none: it costs CPU and
creates false confidence that speaker barge-in works. So these tests synthesize
a realistic echo path (delay + room impulse response + attenuation) and measure
**ERLE** — echo return loss enhancement, the standard metric — rather than
checking that the function returned an array.

The double-talk test is the one that matters most in practice. An AEC that keeps
adapting while the user talks over the assistant destroys its own converged
filter, and the failure mode is that cancellation gets *worse* the longer the
call runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.voice.aec import BLOCK, SAMPLE_RATE, EchoCanceller, erle_db


def _speech_like(n: int, seed: int = 0) -> np.ndarray:
    """A speech-ish signal: a harmonic stack with a wandering envelope.

    Not white noise — an adaptive filter converges trivially on white noise,
    which would flatter the result. Real speech is self-correlated, and that is
    what makes echo cancellation hard.

    The fundamental is derived from the seed, which matters more than it looks:
    an earlier version used the SAME harmonics for every seed and varied only
    the phase, so two "different speakers" were 86% correlated at some lag. A
    64 ms filter finds that lag and cancels the near-end talker legitimately —
    the near-end test then fails for a reason that has nothing to do with the
    canceller. Different fundamentals make the signals genuinely independent,
    the way two real voices are.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 85.0 + (seed * 37) % 130          # 85–215 Hz, a plausible voice range
    sig = np.zeros(n)
    for k in (1, 2, 3, 5, 8):
        f = f0 * k
        sig += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) / f * 100
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t
                             + rng.uniform(0, 3))
    sig *= env
    sig += rng.normal(0, 0.02, n)          # breath noise, decorrelating
    peak = float(np.abs(sig).max()) or 1.0
    return (sig / peak * 0.6).astype(np.float64)


def test_the_test_signals_are_actually_independent():
    """Guards the guard. If two seeds produce correlated signals, the near-end
    test measures the signal generator rather than the canceller."""
    a, b = _speech_like(32000, seed=2), _speech_like(32000, seed=99)
    c = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    c /= np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum())
    assert float(np.abs(c).max()) < 0.35, "near-end is predictable from the reference"


def _echo_path(ref: np.ndarray, *, delay: int, gain: float = 0.5) -> np.ndarray:
    """Delay + a short decaying impulse response + attenuation — a laptop
    speaker bouncing off a desk into the mic."""
    ir = np.array([1.0, 0.0, 0.45, 0.0, 0.0, 0.22, 0.0, 0.0, 0.0, 0.11])
    echoed = np.convolve(ref, ir)[:ref.size]
    out = np.zeros(ref.size)
    if delay < ref.size:
        out[delay:] = echoed[:ref.size - delay]
    return out * gain


def test_it_cancels_a_pure_echo():
    """No near-end speech: the mic hears ONLY the assistant. This is the case
    that has to work before anything else is worth discussing."""
    n = SAMPLE_RATE * 3
    ref = _speech_like(n, seed=1)
    mic = _echo_path(ref, delay=160)          # 10 ms
    out = EchoCanceller().process(mic, ref)
    # Measure on the second half, after the filter has had time to converge.
    half = n // 2
    erle = erle_db(mic[half:], np.asarray(out)[half:])
    assert erle > 8.0, f"only {erle:.1f} dB of echo removed"


def test_near_end_speech_survives():
    """The user's own voice must come through. An AEC that cancels the near-end
    talker is just a mute button."""
    n = SAMPLE_RATE * 2
    ref = _speech_like(n, seed=2)
    near = _speech_like(n, seed=99) * 0.8
    mic = near + _echo_path(ref, delay=120)
    out = np.asarray(EchoCanceller().process(mic, ref), dtype=np.float64)
    # Correlation with the near-end signal must stay high — the voice is intact,
    # not merely "some energy remains".
    half = n // 2
    a, b = out[half:], near[half:]
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.6, f"near-end speech mangled (corr={corr:.2f})"


def test_double_talk_freezes_adaptation():
    """While the user talks over the assistant, the error signal is near-end
    speech, not echo. Adapting on it destroys the converged filter."""
    n = SAMPLE_RATE * 2
    ref = _speech_like(n, seed=3)
    aec = EchoCanceller()
    # Heavy near-end speech throughout ⇒ the freeze must engage repeatedly.
    mic = _speech_like(n, seed=7) * 1.5 + _echo_path(ref, delay=100)
    aec.process(mic, ref)
    assert aec.frozen_blocks > 0, "double-talk detector never engaged"
    assert aec.frozen_blocks < aec.blocks_processed, \
        "adaptation froze for the WHOLE call — the filter can never converge"


def test_delay_is_estimated_not_assumed():
    """Playback reaches the mic tens of ms late. An unaligned filter adapts
    toward noise, so the bulk delay has to be found."""
    n = SAMPLE_RATE * 2
    ref = _speech_like(n, seed=4)
    delay = 480                                # 30 ms
    mic = _echo_path(ref, delay=delay)
    aec = EchoCanceller()
    aec.process(mic, ref)
    assert aec._delay_locked
    # Within a couple of blocks of the truth is enough for the filter to cover.
    assert abs(aec._delay - delay) <= BLOCK * 2, \
        f"estimated {aec._delay}, actual {delay}"


def test_a_longer_echo_delay_still_cancels():
    n = SAMPLE_RATE * 3
    ref = _speech_like(n, seed=5)
    mic = _echo_path(ref, delay=640)           # 40 ms
    out = EchoCanceller().process(mic, ref)
    half = n // 2
    assert erle_db(mic[half:], np.asarray(out)[half:]) > 6.0


# ── Contract / fail-open ────────────────────────────────────────────────────

def test_no_reference_returns_the_mic_untouched():
    mic = _speech_like(1600)
    assert EchoCanceller().process(mic, None) is mic


def test_silence_does_not_divide_by_zero():
    z = np.zeros(SAMPLE_RATE, dtype=np.float64)
    out = np.asarray(EchoCanceller().process(z, z), dtype=np.float64)
    assert np.all(np.isfinite(out))


def test_int16_in_int16_out():
    """The capture path is int16; the canceller must not silently change dtype
    underneath the segmenter."""
    ref = (_speech_like(SAMPLE_RATE) * 32767).astype(np.int16)
    mic = ref.copy()
    out = EchoCanceller().process(mic, ref)
    assert isinstance(out, np.ndarray) and out.dtype == np.int16


def test_bytes_in_bytes_out():
    ref = (_speech_like(SAMPLE_RATE) * 32767).astype("<i2").tobytes()
    out = EchoCanceller().process(ref, ref)
    assert isinstance(out, bytes) and len(out) == len(ref)


def test_output_length_is_preserved_including_a_ragged_tail():
    """Dropping samples would desynchronize the segmenter downstream."""
    n = BLOCK * 5 + 37                       # deliberately not block-aligned
    ref = _speech_like(n)
    out = np.asarray(EchoCanceller().process(ref.copy(), ref), dtype=np.float64)
    assert out.size == n


def test_a_broken_input_fails_open_to_the_mic():
    mic = _speech_like(800)
    assert EchoCanceller().process(mic, "not audio at all") is mic


# ── Seam integration ────────────────────────────────────────────────────────

def test_install_is_a_no_op_while_the_flag_is_off(monkeypatch):
    """Default config must keep today's byte-identical passthrough."""
    from app.core.config_loader import cfg
    from app.voice.aec import install
    monkeypatch.setattr(cfg.voice, "native_aec", False, raising=False)
    assert install() is False


def test_install_registers_into_the_live_seam(monkeypatch):
    """A WRAP of `app/live/aec.py`, not a modification of it (rule L2)."""
    from app.core.config_loader import cfg
    from app.live import aec as seam
    from app.voice.aec import EchoCanceller as EC, install

    monkeypatch.setattr(cfg.voice, "native_aec", True, raising=False)
    monkeypatch.setattr(seam, "_registered", None, raising=False)
    assert install() is True
    assert isinstance(seam.get_aec(), EC)


def test_the_seam_still_fails_open_when_a_processor_raises(monkeypatch):
    from app.core.config_loader import cfg
    from app.live import aec as seam

    class Exploding:
        def process(self, mic, reference=None):
            raise RuntimeError("native module crashed")

    monkeypatch.setattr(cfg.voice, "native_aec", True, raising=False)
    monkeypatch.setattr(seam, "_registered", Exploding(), raising=False)
    mic = np.zeros(64)
    assert seam.process(mic, mic) is mic


@pytest.mark.parametrize("delay", [0, 80, 320])
def test_cancellation_holds_across_plausible_delays(delay):
    n = SAMPLE_RATE * 2
    ref = _speech_like(n, seed=delay + 11)
    mic = _echo_path(ref, delay=delay)
    out = EchoCanceller().process(mic, ref)
    half = n // 2
    assert erle_db(mic[half:], np.asarray(out)[half:]) > 5.0
