"""Acoustic echo cancellation — a real canceller for the seam (design §Noise).

Why this exists
---------------
`app/live/aec.py` has been a typed seam with a no-op passthrough, waiting for a
native module. That was tolerable while the staged engine ran, because the
Windows gap was covered by comparing an incoming transcript against the reply
text we knew we had just spoken.

Under `RealtimeEngine` that guard **cannot work**: there is no locally-known
reply text to compare against in time. So on Windows — where the OS gives no
echo cancellation and `RecordConfig(echoCancel: true)` is a no-op — the mic hears
the assistant, and the model treats its own voice as the user interrupting. Real
cancellation stops being a nice-to-have and becomes the thing that makes speaker
barge-in possible at all.

The algorithm
-------------
Frequency-domain block adaptive filter (overlap-save FDAF), which is the
standard way to run a long echo-path filter cheaply: a time-domain NLMS with
enough taps to cover a room echo (~32 ms = 512 taps at 16 kHz) costs a
512-multiply inner loop per sample and is hopeless in Python, while the same
filter in the frequency domain is one FFT per block.

Three pieces make it work on real audio rather than only on a test signal:

1. **Delay alignment.** Playback reaches the mic tens of milliseconds after we
   emit it (buffering + acoustics). An unaligned filter adapts toward noise. The
   bulk delay is estimated once by cross-correlation and then held.
2. **Double-talk freezing.** When the user speaks *over* the assistant, the
   error signal is dominated by near-end speech. Adapting then actively destroys
   a converged filter — the classic way an AEC gets worse over a call. Adaptation
   freezes whenever near-end energy rises relative to the echo estimate.
3. **Residual suppression.** A linear filter cannot remove non-linear speaker
   distortion. A gentle spectral gate, applied only where the reference is loud,
   takes out what is left.

This is deliberately NOT presented as equivalent to WebRTC's APM. It is a real,
measurable canceller (see `tests/test_voice_aec.py`, which measures ERLE on a
synthetic echo path) that needs no native build. If a native APM is ever wired,
it registers through the same seam and this becomes the fallback.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("zapthetrick.voice.aec")

# 16 kHz mono is the capture format throughout the voice path.
SAMPLE_RATE = 16_000
# Block size. 256 samples = 16 ms — small enough to keep latency invisible,
# large enough that the FFT cost per second stays trivial.
BLOCK = 256
# Filter length in blocks. 4 blocks = 64 ms of echo tail, which covers a laptop
# speaker-to-mic path with room to spare.
PARTITIONS = 4
# NLMS step size. Below ~0.05 convergence is too slow to help a live call; above
# ~0.5 the filter rings on speech.
MU = 0.25
# Regularization, so a silent reference cannot divide by ~0.
EPS = 1e-6
# Double-talk is detected from how much of the MIC energy survives cancellation.
# Converged and echo-only, the residual is a small fraction of the mic; when the
# user talks over us it jumps back toward 1.0 because their voice is not echo.
#
# It must NOT be measured against the echo ESTIMATE: that starts at zero (the
# filter is all zeros), so the detector would fire on block 0 and freeze
# adaptation forever — the filter could never converge and the canceller would
# silently do nothing. That bug is what `frozen_blocks < blocks_processed` in the
# test suite exists to catch.
DOUBLE_TALK_RESIDUAL = 0.55
# Blocks during which adaptation always runs, regardless of the detector. Until
# the filter has some shape, "residual is large" means "not converged yet", not
# "the user is speaking".
WARMUP_BLOCKS = 48          # ~0.75 s at 16 ms/block
# Residual suppression floor: the most a bin may be attenuated. Deliberately
# high — the linear filter already delivers ~20 dB, so this stage only has to
# catch speaker non-linearity, and an aggressive floor eats near-end speech.
RESIDUAL_FLOOR = 0.5
# How strongly the echo estimate pulls a bin's gain down.
RESIDUAL_WEIGHT = 0.3


class EchoCanceller:
    """Overlap-save frequency-domain adaptive filter.

    Stateful across blocks — one instance per session. Not thread-safe; the
    voice path is single-consumer per socket.
    """

    def __init__(self, *, block: int = BLOCK, partitions: int = PARTITIONS,
                 mu: float = MU) -> None:
        self.block = int(block)
        self.partitions = int(partitions)
        self.mu = float(mu)
        self._fft_len = 2 * self.block
        self._bins = self._fft_len // 2 + 1
        # Filter weights, one spectrum per partition.
        self._W = np.zeros((self.partitions, self._bins), dtype=np.complex128)
        # Reference spectra history (most recent first).
        self._X = np.zeros((self.partitions, self._bins), dtype=np.complex128)
        # Overlap-save tails.
        self._ref_tail = np.zeros(self.block, dtype=np.float64)
        self._mic_tail = np.zeros(self.block, dtype=np.float64)
        self._delay = 0
        self._delay_locked = False
        self._frozen_blocks = 0
        self._blocks = 0

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, mic, reference=None):
        """Return `mic` with the echo of `reference` removed.

        Both are int16 numpy arrays or bytes at 16 kHz. Fail-open: any problem
        returns the mic unchanged, because a broken canceller must never take
        the microphone down.
        """
        try:
            m = _as_float(mic)
            if reference is None:
                return mic
            r = _as_float(reference)
            if m.size == 0 or r.size == 0:
                return mic
            if not self._delay_locked:
                self._estimate_delay(m, r)
            r = self._apply_delay(r, m.size)
            out = self._run(m, r)
            return _to_like(out, mic)
        except Exception:  # noqa: BLE001 — never break capture
            log.debug("aec failed; passing mic through", exc_info=True)
            return mic

    @property
    def blocks_processed(self) -> int:
        return self._blocks

    @property
    def frozen_blocks(self) -> int:
        """How many blocks skipped adaptation because of double-talk. A useful
        health signal: near zero during a monologue, high while interrupting."""
        return self._frozen_blocks

    # ── internals ───────────────────────────────────────────────────────────

    def _estimate_delay(self, mic: np.ndarray, ref: np.ndarray) -> None:
        """Coarse bulk delay by cross-correlation, estimated once.

        Re-estimating every block would let a transient lock onto the wrong lag
        and never recover; the acoustic path does not move mid-call.
        """
        n = min(mic.size, ref.size, SAMPLE_RATE // 2)   # <= 500 ms of signal
        if n < self.block * 2:
            return
        a = mic[:n] - mic[:n].mean()
        b = ref[:n] - ref[:n].mean()
        if float(np.sqrt((a * a).sum() * (b * b).sum())) < 1e-9:
            return                                      # silence — nothing to align
        corr = np.abs(np.correlate(a, b, mode="full"))
        # Only a POSITIVE lag is physical: the mic hears playback after we emit
        # it. Search that half only.
        zero = n - 1
        window = corr[zero:zero + SAMPLE_RATE // 4]
        if window.size == 0:
            return
        peak = float(window.max())
        if peak <= 1e-9:
            return
        # Prefer the EARLIEST strong peak, not the global maximum. Voiced speech
        # is quasi-periodic, so the cross-correlation has a comparable peak at
        # every pitch period; taking the argmax can lock onto a later one and
        # misalign the filter by whole periods. The direct acoustic path always
        # arrives first, so the first peak within 90% of the maximum is the
        # physical one.
        candidates = np.flatnonzero(window >= 0.9 * peak)
        lag = int(candidates[0]) if candidates.size else int(np.argmax(window))
        self._delay = int(np.clip(lag, 0, SAMPLE_RATE // 4))
        self._delay_locked = True
        log.debug("aec: locked echo delay at %d samples", self._delay)

    def _apply_delay(self, ref: np.ndarray, want: int) -> np.ndarray:
        if self._delay <= 0:
            out = ref
        else:
            out = np.concatenate([np.zeros(self._delay), ref])
        if out.size < want:
            out = np.concatenate([out, np.zeros(want - out.size)])
        return out[:want]

    def _run(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        n_blocks = mic.size // self.block
        if n_blocks == 0:
            return mic
        out = np.empty(n_blocks * self.block, dtype=np.float64)
        est = np.empty_like(out)            # per-block echo estimate
        talk = np.zeros(n_blocks, dtype=bool)   # blocks judged double-talk

        for i in range(n_blocks):
            lo, hi = i * self.block, (i + 1) * self.block
            m_blk = mic[lo:hi]
            r_blk = ref[lo:hi]

            # Overlap-save: FFT of [previous block | current block].
            x = np.concatenate([self._ref_tail, r_blk])
            self._ref_tail = r_blk.copy()
            X = np.fft.rfft(x)

            # Shift the reference history and insert this block's spectrum.
            self._X[1:] = self._X[:-1]
            self._X[0] = X

            # Echo estimate = sum over partitions of W * X, keep the last block.
            Y = (self._W * self._X).sum(axis=0)
            y_full = np.fft.irfft(Y, n=self._fft_len)
            echo = y_full[self.block:]

            err = m_blk - echo
            out[lo:hi] = err
            est[lo:hi] = echo

            # Double-talk: how much of the mic survived cancellation. Near 0 ⇒
            # we are cancelling well and this is echo-only, so keep adapting.
            # Near 1 ⇒ either the user is talking over us, or we have not
            # converged yet — which is why the warmup below distinguishes them.
            mic_e = float(m_blk @ m_blk)
            err_e = float(err @ err)
            residual = err_e / (mic_e + EPS)
            warming = self._blocks < WARMUP_BLOCKS
            if not warming and residual > DOUBLE_TALK_RESIDUAL:
                self._frozen_blocks += 1
                talk[i] = True
            else:
                # Constrained FDAF: the raw frequency-domain gradient
                # corresponds to a CIRCULAR correlation, which aliases energy
                # into taps the filter does not have. Projecting it back through
                # the time domain and zeroing the second half enforces the
                # linear (non-circular) update. Skipping this constraint is the
                # difference between a filter that converges and one that
                # wanders — it is not an optimization.
                E = np.fft.rfft(np.concatenate([np.zeros(self.block), err]))
                norm = (np.abs(self._X) ** 2).sum(axis=0) + EPS
                grad = np.conj(self._X) * E / norm
                g_time = np.fft.irfft(grad, n=self._fft_len, axis=-1)
                g_time[..., self.block:] = 0.0
                self._W += self.mu * np.fft.rfft(g_time, axis=-1)
            self._blocks += 1

        # Residual suppression on what the linear filter could not remove.
        out = self._suppress_residual(out, est, talk)
        # Any ragged tail passes through unmodified rather than being dropped —
        # losing samples would desynchronize the segmenter downstream.
        if mic.size > out.size:
            out = np.concatenate([out, mic[out.size:]])
        return out

    def _suppress_residual(self, err: np.ndarray, echo_est: np.ndarray,
                           double_talk: np.ndarray) -> np.ndarray:
        """Gentle spectral gate over what the linear filter could not remove.

        Keyed on the ECHO ESTIMATE, not on the raw reference. Keying it on the
        reference attenuates every bin the assistant is loud in — which is
        precisely when the user is interrupting — and measurably destroys the
        near-end voice (correlation fell to 0.03 in the near-end test). The
        estimate, by contrast, is near zero once the linear filter has converged,
        so this stage does almost nothing on clean audio and only bites where
        real echo survives.

        Suppression is also SKIPPED on blocks flagged as double-talk: there the
        residual is the user's speech, and gating it is the one thing this stage
        must never do.
        """
        if err.size < self.block:
            return err
        n = (err.size // self.block) * self.block
        e = err[:n].reshape(-1, self.block)
        y = echo_est[:n].reshape(-1, self.block)
        E = np.fft.rfft(e, axis=1)
        Y = np.fft.rfft(y, axis=1)
        e_mag = np.abs(E)
        y_mag = np.abs(Y)
        # Keep the bin unless the surviving echo estimate dominates it.
        gain = e_mag / (e_mag + RESIDUAL_WEIGHT * y_mag + EPS)
        gain = np.clip(gain, RESIDUAL_FLOOR, 1.0)
        # Near-end speech present ⇒ pass the block through untouched.
        if double_talk.size >= gain.shape[0]:
            gain[double_talk[:gain.shape[0]]] = 1.0
        cleaned = np.fft.irfft(E * gain, n=self.block, axis=1).reshape(-1)
        return np.concatenate([cleaned, err[n:]])


def _as_float(x) -> np.ndarray:
    """int16 / bytes / float array → float64 in roughly [-1, 1]."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(x), dtype="<i2").astype(np.float64) / 32768.0
    a = np.asarray(x)
    if a.dtype == np.int16:
        return a.astype(np.float64) / 32768.0
    return a.astype(np.float64).reshape(-1)


def _to_like(out: np.ndarray, template):
    """Return `out` in the same container/dtype the caller passed in."""
    clipped = np.clip(out, -1.0, 1.0)
    if isinstance(template, (bytes, bytearray, memoryview)):
        return (clipped * 32768.0).astype("<i2").tobytes()
    t = np.asarray(template)
    if t.dtype == np.int16:
        return (clipped * 32768.0).astype(np.int16)
    return clipped.astype(t.dtype if t.dtype.kind == "f" else np.float32)


def install() -> bool:
    """Register this canceller into the shared seam (`app/live/aec.py`).

    A WRAP, not a modification — rule L2: voice-specific behaviour never lands
    inside `app/live/`. Returns whether it was installed; `voice.native_aec`
    off ⇒ the seam keeps its no-op passthrough and behaviour is unchanged.
    """
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.voice, "native_aec", False)):
            return False
        from app.live import aec as seam
        seam.register_aec(EchoCanceller())
        log.info("voice: FDAF echo canceller installed")
        return True
    except Exception:  # noqa: BLE001
        log.warning("voice: could not install the echo canceller", exc_info=True)
        return False


def erle_db(mic: np.ndarray, cleaned: np.ndarray) -> float:
    """Echo Return Loss Enhancement in dB — how much echo energy was removed.

    The standard AEC metric, exposed so the effect is measurable rather than
    asserted. Higher is better; 0 means nothing was cancelled.
    """
    m = _as_float(mic)
    c = _as_float(cleaned)
    n = min(m.size, c.size)
    num = float((m[:n] ** 2).sum())
    den = float((c[:n] ** 2).sum()) + 1e-12
    if num <= 1e-12:
        return 0.0
    return float(10.0 * np.log10(num / den))


__all__ = ["EchoCanceller", "install", "erle_db", "SAMPLE_RATE", "BLOCK"]
