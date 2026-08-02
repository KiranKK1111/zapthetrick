"""
DEPRECATED — superseded by `app/voice/engine.py` (realtime-voice-mode §2).

This turn-based seam (`turn(pcm16) -> AsyncIterator[(text, audio)]`) cannot host
a speech-native model: it presupposes that somebody else already decided where
the utterance ended — precisely the decision a realtime model makes better — and
it has no channel for the model to say "the user started talking, I stopped", so
native barge-in is inexpressible. `app.voice.engine.VoiceSession` replaces it
with a SESSION-scoped contract where turn-taking authority moves with the engine.

Nothing depends on this module today: no engine was ever registered, so
`get_s2s()` has always returned None and the staged path has always run. It is
kept as a shim for one release so a stray import stays safe, and is deleted once
the migration completes. `cfg.voice.s2s_engine` is likewise still honoured —
`app.voice.policy.configured_engine()` maps "omni" onto the new
`voice.engine: realtime`.

This file lives under `app/live/`, so reducing it to a shim is the ONE edit the
voice design makes inside that package. It removes dead code, changes no Live
behaviour, and is covered by the existing Live suite. Recorded explicitly rather
than left implicit.

---

Original docstring follows.

Speech-to-speech engine seam (Flow 1, 2026-07-28).

The voice surface's duplex contract is: ONE WebSocket, audio (or text) up,
`audio` frames (JSON meta + binary MP3) down, with generation-numbered barge
cancellation. TODAY that contract is served by the STAGED engine — segmenter →
STT → the FE-driven agent turn → on-pod Kokoro synthesis streamed back over the
socket (see `/ws/voice` speak frames in routes_ws.py).

This module is the drop-in point for a TRUE speech-native model (Qwen-Omni /
Moshi class): register an engine that consumes the utterance audio directly and
yields synthesized reply audio, and the SAME WebSocket frames carry its output —
the FE never changes. Until one is registered (it needs on-hardware validation
and a VRAM budget decision), `get_s2s()` returns None and the staged path runs.

Fail-open everywhere, mirroring the kokoro seam.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol

from app.core.config_loader import cfg


class SpeechToSpeechEngine(Protocol):
    """Contract a speech-native engine implements: one user utterance's PCM in,
    a stream of (transcript_delta, mp3_audio_chunk) out. Either element of the
    tuple may be empty for a given yield."""

    def turn(self, pcm16: bytes, *, context: dict
             ) -> AsyncIterator[tuple[str, bytes]]:
        ...


_registered: "SpeechToSpeechEngine | None" = None


def register_s2s(engine: "SpeechToSpeechEngine") -> None:
    """A pod build with a speech-native model calls this at startup."""
    global _registered
    _registered = engine


def enabled() -> bool:
    return (getattr(cfg.voice, "s2s_engine", "staged") or "staged") == "omni"


def get_s2s() -> "SpeechToSpeechEngine | None":
    """The active speech-native engine, or None → the staged pipeline serves
    the duplex contract (today's default). Never raises."""
    try:
        if enabled() and _registered is not None:
            return _registered
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = ["SpeechToSpeechEngine", "register_s2s", "get_s2s", "enabled"]
