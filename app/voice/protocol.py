"""Voice wire protocol v2 — self-describing frames (design §5).

One module owns encode AND decode for both ends so the layout cannot drift.

The defect this replaces
------------------------
v1 sent a JSON meta frame ``{"type":"audio","seq","gen"}`` immediately followed
by a bare binary frame, and the client paired them **by list position**
(`_audioMeta.removeAt(0)`). Bytes already written to the socket cannot be
unsent, so a stale binary arriving after a barge-in paired with the *next*
turn's meta and the list stayed off-by-one for the rest of the session.

v2 makes a binary frame interpretable **on its own**:

    ┌────────┬──────┬──────┬────────┬────────┬───────────┐
    │ magic  │ ver  │ kind │ seq    │ gen    │ payload   │
    │ 2B     │ 1B   │ 1B   │ 4B u32 │ 4B u32 │ N bytes   │
    └────────┴──────┴──────┴────────┴────────┴───────────┘
    all integers little-endian

Because ``gen`` travels *inside the frame*, a frame from a superseded
generation is dropped **by value**. There is no pairing step, so there is no
list to fall out of alignment, and bytes in flight during a barge-in are
handled correctly by construction rather than by bookkeeping.

``seq`` is monotonic within a turn, so gaps are detectable: a missing chunk is
reported instead of silently stalling the phase machine.

Everything here is pure and dependency-free so both the server and the test
suite can exercise it without a socket.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# 'ZV' — ZapTheTrick Voice. Guards against a stray non-protocol binary frame
# (e.g. an old client, or a proxy injecting something) being decoded as audio.
MAGIC = b"ZV"
VERSION = 2

# magic(2) + ver(1) + kind(1) + seq(4) + gen(4)
HEADER_LEN = 12
_HEADER = struct.Struct("<2sBBII")


class FrameKind(IntEnum):
    """Binary payload type. Values are wire constants — never renumber."""

    AUDIO_PCM = 1      # int16 mono PCM at SAMPLE_RATE
    AUDIO_MP3 = 2      # encoded MP3 (what tts_synth returns today)
    RESERVED = 3


# The sample rate PCM frames are expected at, both directions. Matches the
# segmenter's expectation so no resampling happens on the staged path.
SAMPLE_RATE = 16_000


class ProtocolError(ValueError):
    """A binary frame that is not decodable as protocol v2."""


@dataclass(frozen=True)
class AudioFrame:
    """A decoded binary frame. `gen` is the generation it was produced under —
    compare against the client's current floor to decide whether to render."""

    kind: FrameKind
    seq: int
    gen: int
    payload: bytes

    @property
    def is_audio(self) -> bool:
        return self.kind in (FrameKind.AUDIO_PCM, FrameKind.AUDIO_MP3)


def encode_audio(payload: bytes, *, kind: FrameKind, seq: int,
                 gen: int) -> bytes:
    """Build one self-describing binary frame.

    `seq` and `gen` are unsigned 32-bit; they are masked rather than rejected so
    a long-running session cannot raise mid-turn (wrap-around is harmless — the
    client compares against a floor that wraps identically).
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ProtocolError(f"payload must be bytes, got {type(payload)!r}")
    head = _HEADER.pack(MAGIC, VERSION, int(kind), seq & 0xFFFFFFFF,
                        gen & 0xFFFFFFFF)
    return head + bytes(payload)


def decode_audio(raw: bytes) -> AudioFrame:
    """Parse a binary frame. Raises `ProtocolError` on anything malformed.

    A truncated header, a bad magic, an unknown version or an unknown kind are
    all errors rather than best-effort guesses: silently misinterpreting a frame
    is exactly the failure mode v2 exists to remove. An EMPTY payload is legal
    (a zero-length chunk is still a real, orderable event).
    """
    if len(raw) < HEADER_LEN:
        raise ProtocolError(
            f"frame shorter than header ({len(raw)} < {HEADER_LEN})")
    magic, ver, kind, seq, gen = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ProtocolError(f"unsupported version {ver}")
    try:
        k = FrameKind(kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown frame kind {kind}") from exc
    return AudioFrame(kind=k, seq=seq, gen=gen, payload=raw[HEADER_LEN:])


def is_protocol_frame(raw: bytes) -> bool:
    """Cheap sniff used by the client to tell a v2 frame from a legacy bare
    binary during the transition. Never raises."""
    return len(raw) >= HEADER_LEN and raw[:2] == MAGIC and raw[2] == VERSION


# ── Control frames (JSON) ────────────────────────────────────────────────────
# Down-frames the client renders. Kept as builders (not raw dicts at call sites)
# so a field rename is one edit and the shapes stay uniform.

def session_ready(engine: str, voice: str, *, turn_detection: str,
                  protocol: int = VERSION) -> dict:
    return {"type": "session.ready", "engine": engine, "voice": voice,
            "turn_detection": turn_detection, "protocol": protocol}


def transcript(role: str, text: str, *, final: bool, seq: int = 0,
               complete: bool = False) -> dict:
    """`complete` is the semantic "this partial reads as a finished thought"
    flag that drives client-side speculation. Meaningless on a final."""
    f = {"type": "transcript", "role": role, "text": text, "final": final,
         "seq": seq}
    if not final:
        f["complete"] = bool(complete)
    return f


def phase(value: str) -> dict:
    """listening | thinking | speaking | tool | error"""
    return {"type": "phase", "value": value}


def tool(id: str, name: str, state: str) -> dict:
    """state: start | ok | fail"""
    return {"type": "tool", "id": id, "name": name, "state": state}


def usage(input_tokens: int, output_tokens: int, *,
          spent_usd: float | None = None,
          ceiling_pct: float | None = None) -> dict:
    f = {"type": "usage", "input_tokens": int(input_tokens),
         "output_tokens": int(output_tokens)}
    if spent_usd is not None:
        f["spent_usd"] = round(float(spent_usd), 4)
    if ceiling_pct is not None:
        f["ceiling_pct"] = round(float(ceiling_pct), 3)
    return f


def engine_switch(from_: str, to: str, reason: str) -> dict:
    return {"type": "engine_switch", "from": from_, "to": to, "reason": reason}


def generation(n: int) -> dict:
    """New floor — the client discards any audio frame whose `gen` is below."""
    return {"type": "generation", "n": int(n)}


def turn_complete(user: str, assistant: str, *, interrupted: bool = False,
                  chunks: int = 0) -> dict:
    """Exactly one per turn (design Property 3). `chunks` is how many audio
    frames the engine emitted, so the client can assert it saw them all."""
    return {"type": "turn_complete", "user": user, "assistant": assistant,
            "interrupted": bool(interrupted), "chunks": int(chunks)}


def dropped(seq: int, reason: str) -> dict:
    """Property 1: audio the engine abandoned is REPORTED, never silently lost."""
    return {"type": "dropped", "seq": int(seq), "reason": reason}


def error(kind: str, detail: str, *, recoverable: bool = True) -> dict:
    return {"type": "error", "kind": kind, "detail": detail,
            "recoverable": bool(recoverable)}


def stt_status(state: str, detail: str) -> dict:
    """Retained from v1 — a dead mic stays explainable."""
    return {"type": "stt_status", "state": state, "detail": detail}


__all__ = [
    "MAGIC", "VERSION", "HEADER_LEN", "SAMPLE_RATE", "FrameKind", "AudioFrame",
    "ProtocolError", "encode_audio", "decode_audio", "is_protocol_frame",
    "session_ready", "transcript", "phase", "tool", "usage", "engine_switch",
    "generation", "turn_complete", "dropped", "error", "stt_status",
]
