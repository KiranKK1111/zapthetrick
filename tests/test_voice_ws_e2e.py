"""END-TO-END contract test for the chat voice surface (`/ws/voice`).

Drives the REAL WebSocket handler + the REAL AudioStreamSegmenter + the real
turn gates over synthetic PCM — only the VAD (audio-time double) and STT
(scripted transcripts) are mocked. This is the closest software can get to a
live mic session; what it cannot cover (real STT accuracy, speaker echo, AEC)
needs an on-hardware session and is documented in memory.

Locks the whole ChatGPT-style voice contract in one place:
  • `ready` on connect;
  • streaming `partial` frames carrying the semantic `complete` flag that
    drives client-side speculation (doc §6 Incremental Thinking);
  • silence → a finalized `transcript` (server-side turn detection);
  • the first-real-word gate (a lone "uh" takes no turn);
  • STT failure surfacing (`stt_status`, never silence);
  • while `assistant_speaking`: semantic barge classification on transcripts
    ("yeah" → backchannel = keep talking; "wait" → interrupt = stop).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import stream as stream_mod
from app.core.config_loader import cfg

SR = 16000


class FakeStreamingVAD:
    """Audio-time VAD double (same convention as test_adaptive_endpointing):
    a chunk whose first sample is 1.0 counts as voiced, 0.0 as silence."""

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


class ScriptedStt:
    """Finals + partials come from scripts, in order."""

    def __init__(self, finals, partials=None):
        self.finals = list(finals)
        self.partials = list(partials or [])

    async def final(self, audio, prompt=None):
        if not self.finals:
            return "", None
        nxt = self.finals.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt, 0.9

    async def partial(self, audio):
        return self.partials.pop(0) if self.partials else ""


@pytest.fixture
def voice_ws(monkeypatch):
    """A connected /ws/voice with pinned timings + scripted STT. Yields
    (websocket, stt, push) where push(voiced, ms) streams synthetic PCM."""
    monkeypatch.setattr(stream_mod.vad, "StreamingVAD", FakeStreamingVAD)
    # Pod-like timings, pinned so the test is config-independent.
    monkeypatch.setattr(cfg.audio, "endpoint_silence_ms", 550, raising=False)
    monkeypatch.setattr(cfg.audio, "min_utterance_ms", 100, raising=False)
    monkeypatch.setattr(cfg.audio, "short_utterance_gap_ms", 400, raising=False)
    monkeypatch.setattr(cfg.audio, "partial_min_ms", 400, raising=False)
    monkeypatch.setattr(cfg.audio, "partial_interval_ms", 100, raising=False)
    monkeypatch.setattr(cfg.audio, "prosody_endpointing", False, raising=False)
    # The final pass must run the scripted FINAL (not adopt the partial's text),
    # regardless of how the local config pairs partial/final providers.
    monkeypatch.setattr(cfg.stt, "final_from_partial", False, raising=False)
    # No network: the speech-start provider warm is a no-op.
    import app.perceived.prefetch as prefetch

    async def _no_warm(*a, **k):
        return None

    monkeypatch.setattr(prefetch, "warm_live_provider", _no_warm)
    # Tokenless test socket: neutralize WS auth (the dev box enforces native
    # auth, which would close the socket pre-accept with 1008) and skip the
    # device-user DB lookup (no lifespan → no DB in this harness).
    import app.api.auth as auth_mod

    monkeypatch.setattr(auth_mod, "authenticate_ws",
                        lambda tok, **k: (None, None))

    async def _no_uid(uid):
        return None

    monkeypatch.setattr(auth_mod, "ws_user_id", _no_uid)

    stt = ScriptedStt(finals=[])
    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_with_confidence",
                        stt.final)
    monkeypatch.setattr(stream_mod.stt_factory, "transcribe_partial",
                        stt.partial)

    from app.main import app
    client = TestClient(app)  # no `with` → lifespan (DB etc.) never runs
    with client.websocket_connect("/ws/voice") as ws:

        def push(voiced: bool, ms: int):
            n = int(SR * ms / 1000)
            pcm = np.full(n, 32000 if voiced else 0, dtype="<i2")
            ws.send_bytes(pcm.tobytes())

        assert ws.receive_json()["type"] == "ready"
        yield ws, stt, push


def test_partial_carries_the_semantic_complete_flag(voice_ws):
    ws, stt, push = voice_ws
    # A partial that READS complete → complete: true (speculation may start).
    stt.partials = ["what is a mutex"]
    stt.finals = ["what is a mutex?"]
    push(True, 600)                      # ≥ partial_min_ms → one partial pass
    frame = ws.receive_json()
    assert frame["type"] == "partial"
    assert frame["text"] == "what is a mutex"
    assert frame["complete"] is True
    # End-of-turn: 550*0.55≈303ms suffices for a complete partial.
    push(False, 400)
    final = ws.receive_json()
    assert final["type"] == "transcript"
    assert final["text"] == "what is a mutex?"


def test_incomplete_partial_is_flagged_not_speculated(voice_ws):
    ws, stt, push = voice_ws
    stt.partials = ["can you explain"]
    stt.finals = ["can you explain virtual memory"]
    push(True, 600)
    frame = ws.receive_json()
    assert frame["type"] == "partial"
    assert frame["complete"] is False    # dangling stem → no speculation
    # An incomplete tail demands the LONG gap (1200ms) — the "thinking pause"
    # is respected, then the finished thought arrives as one transcript.
    push(False, 1300)
    assert ws.receive_json()["type"] == "transcript"


def test_lone_filler_takes_no_turn(voice_ws):
    ws, stt, push = voice_ws
    # "uh" (a cough the recognizer rendered as a token) → NO transcript frame.
    stt.finals = ["uh", "what is recursion"]
    push(True, 200)                      # short burst, no partial fires
    push(False, 600)                     # finalizes → gated, nothing sent
    # The NEXT real utterance must be the next frame — proving the filler
    # produced none.
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript"
    assert frame["text"] == "what is recursion"


def test_stt_failure_is_surfaced_never_silent(voice_ws):
    ws, stt, push = voice_ws
    stt.finals = [RuntimeError("model exploded")]
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "stt_status"
    assert frame["state"] in ("error", "empty")


def _embedder_ready() -> bool:
    try:
        from app.rag import embedder
        embedder.embed(["warm"])
        return bool(embedder.is_ready())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _embedder_ready(), reason="embedder unavailable")
def test_barge_classification_while_assistant_speaks(voice_ws):
    ws, stt, push = voice_ws
    ws.send_json({"type": "assistant_speaking", "value": True})
    # Backchannel — the user is following along; the client keeps speaking.
    stt.finals = ["yeah", "wait", "what is recursion"]
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript" and frame["text"] == "yeah"
    assert frame.get("barge") == "backchannel"
    # Interrupt cue — the client must stop playback.
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript" and frame["text"] == "wait"
    assert frame.get("barge") == "interrupt"
    # Speaking off → transcripts carry NO barge field (zero added latency).
    ws.send_json({"type": "assistant_speaking", "value": False})
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript"
    assert "barge" not in frame


def test_duplex_speak_streams_audio_and_barge_cancels(voice_ws, monkeypatch):
    """Flow 1 contract: speak chunks come back as audio meta + binary MP3 over
    the SAME socket, in order; stop_speaking (barge) bumps the generation and
    silences everything queued from the interrupted reply."""
    ws, stt, push = voice_ws
    import app.live.tts_synth as tts_synth

    async def fake_synth(text, voice_id=None, *, speed=1.0):
        return b"MP3:" + text.encode()

    monkeypatch.setattr(tts_synth, "synthesize", fake_synth)

    ws.send_json({"type": "speak", "seq": 1, "text": "hello there"})
    meta = ws.receive_json()
    assert meta["type"] == "audio" and meta["seq"] == 1
    assert ws.receive_bytes() == b"MP3:hello there"

    ws.send_json({"type": "speak", "seq": 2, "text": "second sentence"})
    meta = ws.receive_json()
    assert meta["type"] == "audio" and meta["seq"] == 2
    assert ws.receive_bytes() == b"MP3:second sentence"

    # Barge: everything after the stop must be from the NEW generation only.
    ws.send_json({"type": "stop_speaking"})
    frame = ws.receive_json()
    assert frame["type"] == "speech_stopped"
    new_gen = frame["gen"]
    ws.send_json({"type": "speak", "seq": 3, "text": "fresh reply"})
    meta = ws.receive_json()
    assert meta["type"] == "audio" and meta["seq"] == 3
    assert meta["gen"] == new_gen
    assert ws.receive_bytes() == b"MP3:fresh reply"


def test_duplex_synth_failure_reports_speak_error(voice_ws, monkeypatch):
    ws, stt, push = voice_ws
    import app.live.tts_synth as tts_synth

    async def broken_synth(text, voice_id=None, *, speed=1.0):
        return b""

    monkeypatch.setattr(tts_synth, "synthesize", broken_synth)
    ws.send_json({"type": "speak", "seq": 7, "text": "will fail"})
    frame = ws.receive_json()
    assert frame["type"] == "speak_error" and frame["seq"] == 7
