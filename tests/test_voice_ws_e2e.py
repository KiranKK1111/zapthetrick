"""END-TO-END contract test for the chat voice surface (`/ws/voice`).

Drives the REAL WebSocket handler + the REAL AudioStreamSegmenter + the real
turn gates over synthetic PCM — only the VAD (audio-time double) and STT
(scripted transcripts) are mocked. This is the closest software can get to a
live mic session; what it cannot cover (real STT accuracy, speaker echo, AEC)
needs an on-hardware session and is documented in memory.

Locks the whole ChatGPT-style voice contract in one place, on wire protocol v2
(`app/voice/protocol.py`):
  • `session.ready` + `generation` + `phase` on connect;
  • streaming non-final `transcript` frames carrying the semantic `complete`
    flag that drives client-side speculation (doc §6 Incremental Thinking);
  • silence → a FINAL `transcript` (server-side turn detection);
  • the first-real-word gate (a lone "uh" takes no turn);
  • STT failure surfacing (`stt_status`, never silence);
  • while the assistant speaks: semantic barge classification decides whether a
    transcript interrupts ("wait") or is a backchannel ("yeah");
  • duplex synthesis as SELF-DESCRIBING binary frames, in emission order, with a
    barge-in bumping the generation so stale audio is dropped by value.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import stream as stream_mod
from app.core.config_loader import cfg
from app.voice import protocol as P

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

        ready = ws.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["engine"] == "staged"        # default config, zero spend
        assert ready["protocol"] == 2
        assert ws.receive_json()["type"] == "generation"
        assert ws.receive_json() == {"type": "phase", "value": "listening"}
        yield ws, stt, push


def test_partial_carries_the_semantic_complete_flag(voice_ws):
    ws, stt, push = voice_ws
    # A partial that READS complete → complete: true (speculation may start).
    stt.partials = ["what is a mutex"]
    stt.finals = ["what is a mutex?"]
    push(True, 600)                      # ≥ partial_min_ms → one partial pass
    frame = ws.receive_json()
    assert frame["type"] == "transcript" and frame["final"] is False
    assert frame["text"] == "what is a mutex"
    assert frame["complete"] is True
    # End-of-turn: 550*0.55≈303ms suffices for a complete partial.
    push(False, 400)
    final = ws.receive_json()
    assert final["type"] == "transcript" and final["final"] is True
    assert final["text"] == "what is a mutex?"


def test_incomplete_partial_is_flagged_not_speculated(voice_ws):
    ws, stt, push = voice_ws
    stt.partials = ["can you explain"]
    stt.finals = ["can you explain virtual memory"]
    push(True, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript" and frame["final"] is False
    assert frame["complete"] is False    # dangling stem → no speculation
    # An incomplete tail demands the LONG gap (1200ms) — the "thinking pause"
    # is respected, then the finished thought arrives as one FINAL transcript.
    push(False, 1300)
    late = ws.receive_json()
    assert late["type"] == "transcript" and late["final"] is True


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
    assert frame["type"] == "transcript" and frame["final"] is True
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
    """While the assistant speaks, a transcript is classified interrupt-vs-
    backchannel BEFORE it can take the floor (Requirement 3.6).

    Under v2 the verdict is acted on server-side rather than annotated onto the
    frame: a backchannel yields a transcript and nothing else, an interrupt
    additionally bumps the generation so in-flight audio is dropped by value.
    """
    ws, stt, push = voice_ws
    ws.send_json({"type": "assistant_speaking", "value": True})

    # Backchannel — the user is following along; no generation bump.
    stt.finals = ["yeah", "wait"]
    push(True, 200)
    push(False, 600)
    frame = ws.receive_json()
    assert frame["type"] == "transcript" and frame["text"] == "yeah"

    # Interrupt cue — the floor moves, so a new generation is published FIRST,
    # before the utterance is allowed to take the turn.
    push(True, 200)
    push(False, 600)
    gen = _until(ws, "generation")
    assert gen["n"] >= 1
    assert _until(ws, "transcript")["text"] == "wait"


def _until(ws, kind: str, limit: int = 6) -> dict:
    """Read JSON frames until one of `kind` arrives.

    Stops AT the match rather than reading a fixed count — over-reading a socket
    that has nothing further to send blocks the test forever, which is exactly
    the class of stall this feature exists to remove.
    """
    seen = []
    for _ in range(limit):
        frame = ws.receive_json()
        seen.append(frame["type"])
        if frame["type"] == kind:
            return frame
    raise AssertionError(f"no {kind!r} frame within {limit}; saw {seen}")


def _collect(ws, until: str = "turn_complete", limit: int = 24):
    """Read mixed JSON/binary frames until `until` arrives.

    Returns (json_frames, decoded_audio_frames). Deliberately does NOT assert a
    particular interleaving of transcript and audio: the engine guarantees AUDIO
    order, not the relative order of a transcript against a chunk that finished
    synthesizing concurrently. Asserting the interleaving would be testing a
    scheduling accident.
    """
    js, audio = [], []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio.append(P.decode_audio(msg["bytes"]))
            continue
        frame = json.loads(msg["text"])
        js.append(frame)
        if frame.get("type") == until:
            return js, audio
    raise AssertionError(f"no {until!r} within {limit} frames; "
                         f"saw {[f.get('type') for f in js]}")


def test_duplex_speak_streams_ordered_self_describing_audio(voice_ws,
                                                            monkeypatch):
    """Duplex contract on v2: reply chunks come back as SELF-DESCRIBING binary
    frames over the same socket, in EMISSION ORDER, each carrying its own seq
    and generation — so there is no meta/binary pairing to fall out of step.

    Chunk 0 is made slow and chunk 1 fast, so the reorder buffer is actually
    exercised: without it, 1 would overtake 0 (defect 4).
    """
    ws, stt, push = voice_ws
    import app.live.tts_synth as tts_synth
    calls = []

    async def fake_synth(text, voice_id=None, *, speed=1.0):
        calls.append((text, voice_id, speed))
        if text == "hello there":
            await asyncio.sleep(0.25)        # the slow primary / Edge fallback
        return b"MP3:" + text.encode()

    monkeypatch.setattr(tts_synth, "synthesize", fake_synth)

    # The user's CHOSEN voice + speed must reach synthesis (the wrong-voice bug).
    ws.send_json({"type": "speak", "seq": 0, "text": "hello there",
                  "voice": "nova", "speed": 1.25})
    ws.send_json({"type": "speak", "seq": 1, "text": "second sentence"})
    ws.send_json({"type": "reply_end", "chunks": 2})

    js, audio = _collect(ws)

    # Property 2 — emission order, even though chunk 1 synthesized first.
    assert [a.seq for a in audio] == [0, 1]
    assert [a.payload for a in audio] == [b"MP3:hello there",
                                          b"MP3:second sentence"]
    assert audio[0].gen == audio[1].gen          # one turn, one generation
    assert calls[0] == ("hello there", "nova", 1.25)

    said = [f["text"] for f in js
            if f["type"] == "transcript" and f["role"] == "assistant"]
    assert said == ["hello there", "second sentence"]

    # Property 3 — exactly one turn-complete, carrying the chunk count.
    done = [f for f in js if f["type"] == "turn_complete"]
    assert len(done) == 1 and done[0]["chunks"] == 2


def test_barge_in_publishes_a_new_generation_floor(voice_ws, monkeypatch):
    """`stop_speaking` bumps the generation. Anything synthesized for the old
    one is unrenderable BY VALUE, because the frame carries its own gen."""
    ws, stt, push = voice_ws
    import app.live.tts_synth as tts_synth

    async def fake_synth(text, voice_id=None, *, speed=1.0):
        return b"MP3:" + text.encode()

    monkeypatch.setattr(tts_synth, "synthesize", fake_synth)

    ws.send_json({"type": "speak", "seq": 0, "text": "interrupt me"})
    ws.send_json({"type": "reply_end", "chunks": 1})
    ws.receive_json()                                   # assistant transcript
    ws.receive_json()                                   # phase: speaking
    before = P.decode_audio(ws.receive_bytes())
    ws.receive_json()                                   # turn_complete
    ws.receive_json()                                   # phase: listening

    ws.send_json({"type": "stop_speaking"})
    gen = _until(ws, "generation")
    assert gen["n"] > before.gen

    # The next reply is tagged with the NEW floor, so a client holding it
    # renders this and rejects anything still in flight from the old one.
    ws.send_json({"type": "speak", "seq": 0, "text": "fresh reply"})
    ws.send_json({"type": "reply_end", "chunks": 1})
    ws.receive_json()                                   # assistant transcript
    ws.receive_json()                                   # phase: speaking
    after = P.decode_audio(ws.receive_bytes())
    assert after.payload == b"MP3:fresh reply"
    assert after.gen == gen["n"]


def test_duplex_synth_failure_is_reported_not_silent(voice_ws, monkeypatch):
    """Property 1 — abandoned audio is REPORTED. Under v1 an unpaired frame
    left the client's watchdog armed for a full 15 s; now the turn advances."""
    ws, stt, push = voice_ws
    import app.live.tts_synth as tts_synth

    async def broken_synth(text, voice_id=None, *, speed=1.0):
        return b""

    monkeypatch.setattr(tts_synth, "synthesize", broken_synth)
    ws.send_json({"type": "speak", "seq": 0, "text": "will fail"})
    ws.send_json({"type": "reply_end", "chunks": 1})
    ws.receive_json()                                   # assistant transcript
    drop = ws.receive_json()
    assert drop["type"] == "dropped" and drop["seq"] == 0
    assert "empty synthesis" in drop["reason"]
    # …and the turn still terminates rather than hanging (Property 3).
    assert ws.receive_json()["type"] == "turn_complete"
