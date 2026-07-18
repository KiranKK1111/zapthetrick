"""VAD degradation: when Silero/torch is unavailable, `has_speech` must fall
back to a dependency-free energy gate instead of raising — otherwise an
exception propagates out of the live WebSocket receive loop and tears the
socket down (the "live audio keeps failing on the server" reconnect loop)."""
from __future__ import annotations

import numpy as np

from app.audio import vad


def test_energy_gate_distinguishes_loud_from_silence():
    assert vad._energy_has_speech(np.zeros(512, dtype=np.float32)) is False
    loud = np.ones(512, dtype=np.float32) * 0.3
    assert vad._energy_has_speech(loud) is True


def test_has_speech_falls_back_when_silero_raises(monkeypatch):
    monkeypatch.setattr(vad, "_silero_failed", False)

    def _boom(_audio):
        raise RuntimeError("torch not installed")

    monkeypatch.setattr(vad, "_silero_has_speech", _boom)

    loud = np.ones(512, dtype=np.float32) * 0.3
    silence = np.zeros(512, dtype=np.float32)
    # Must NOT raise — degrades to the energy gate.
    assert vad.has_speech(loud) is True
    assert vad.has_speech(silence) is False
    # And it latched the failure so it won't keep retrying the heavy path.
    assert vad._silero_failed is True
