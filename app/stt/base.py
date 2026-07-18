"""
STT provider interface.

Audio in, text out. Concrete providers (`faster_whisper`, future
`deepgram`, `assemblyai`) implement `transcribe(audio_np) -> str`. The
factory picks one based on `cfg.stt.provider` so callers stay decoupled.
"""
from __future__ import annotations

from typing import Protocol


class STT(Protocol):
    """A speech-to-text engine that turns a numpy audio array into text."""

    def transcribe(self, audio_np: "object") -> str:  # numpy.ndarray at runtime
        ...
