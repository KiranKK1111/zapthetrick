"""Wire protocol v2 (design §5, Properties 1/2/4).

The adversarial ordering test is the direct regression test for defect 1 — the
off-by-one that made a stale binary frame pair with the next turn's metadata and
stay misaligned for the rest of the session.
"""
from __future__ import annotations

import pytest

from app.voice import protocol as P


# ── Round trip ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [P.FrameKind.AUDIO_PCM, P.FrameKind.AUDIO_MP3])
@pytest.mark.parametrize("payload", [b"", b"\x00", b"abc" * 500])
def test_round_trip(kind, payload):
    frame = P.encode_audio(payload, kind=kind, seq=42, gen=7)
    out = P.decode_audio(frame)
    assert (out.kind, out.seq, out.gen, out.payload) == (kind, 42, 7, payload)


def test_empty_payload_is_legal():
    """A zero-length chunk is still a real, orderable event — it must decode,
    not raise, or a gap appears in the sequence for no reason."""
    out = P.decode_audio(P.encode_audio(b"", kind=P.FrameKind.AUDIO_PCM,
                                        seq=0, gen=0))
    assert out.payload == b"" and out.is_audio


def test_header_is_fixed_width():
    frame = P.encode_audio(b"xyz", kind=P.FrameKind.AUDIO_MP3, seq=1, gen=1)
    assert len(frame) == P.HEADER_LEN + 3


def test_large_seq_and_gen_wrap_rather_than_raise():
    """A long session must never raise mid-turn on counter width."""
    frame = P.encode_audio(b"a", kind=P.FrameKind.AUDIO_PCM,
                           seq=2 ** 32 + 5, gen=2 ** 32 + 1)
    out = P.decode_audio(frame)
    assert out.seq == 5 and out.gen == 1


# ── Malformed input is an error, never a guess ──────────────────────────────

def test_truncated_header_raises():
    with pytest.raises(P.ProtocolError):
        P.decode_audio(b"ZV\x02\x01")


def test_bad_magic_raises():
    good = P.encode_audio(b"a", kind=P.FrameKind.AUDIO_MP3, seq=0, gen=0)
    with pytest.raises(P.ProtocolError):
        P.decode_audio(b"XX" + good[2:])


def test_unknown_version_raises():
    good = bytearray(P.encode_audio(b"a", kind=P.FrameKind.AUDIO_MP3,
                                    seq=0, gen=0))
    good[2] = 99
    with pytest.raises(P.ProtocolError):
        P.decode_audio(bytes(good))


def test_unknown_kind_raises():
    good = bytearray(P.encode_audio(b"a", kind=P.FrameKind.AUDIO_MP3,
                                    seq=0, gen=0))
    good[3] = 77
    with pytest.raises(P.ProtocolError):
        P.decode_audio(bytes(good))


def test_sniffer_rejects_a_legacy_bare_binary():
    assert not P.is_protocol_frame(b"\xff\xfb\x90raw mp3 bytes")
    assert P.is_protocol_frame(
        P.encode_audio(b"x", kind=P.FrameKind.AUDIO_MP3, seq=0, gen=0))


# ── Property 4: stale audio is unrepresentable, not merely filtered ─────────

def test_stale_frame_from_a_superseded_generation_is_droppable_by_value():
    """The regression test for defect 1.

    A chunk written before a barge-in cannot be unsent. Under v1 the client
    paired it positionally with the NEXT turn's metadata and stayed off-by-one
    forever. Under v2 the frame carries its own generation, so a client holding
    a floor of 1 rejects it by inspecting the frame alone — no queue state, no
    arrival-order assumption.
    """
    in_flight = P.encode_audio(b"old", kind=P.FrameKind.AUDIO_MP3,
                               seq=3, gen=0)
    next_turn = P.encode_audio(b"new", kind=P.FrameKind.AUDIO_MP3,
                               seq=0, gen=1)

    floor = 1                       # client bumped its generation on barge-in
    rendered = [f.payload for f in map(P.decode_audio, [in_flight, next_turn])
                if f.gen >= floor]

    assert rendered == [b"new"]


def test_ordering_is_recoverable_from_the_frames_alone():
    """Property 2: a client that receives chunks out of order can still render
    them in emission order, because `seq` lives in the frame rather than in a
    side list that a fallback path could append to late."""
    frames = [P.encode_audio(str(i).encode(), kind=P.FrameKind.AUDIO_MP3,
                             seq=i, gen=0) for i in range(5)]
    shuffled = [frames[3], frames[0], frames[4], frames[1], frames[2]]
    decoded = sorted(map(P.decode_audio, shuffled), key=lambda f: f.seq)
    assert [f.payload for f in decoded] == [b"0", b"1", b"2", b"3", b"4"]


def test_sequence_gaps_are_detectable():
    """A missing chunk must be reportable rather than a silent stall."""
    seqs = [P.decode_audio(P.encode_audio(b"x", kind=P.FrameKind.AUDIO_MP3,
                                          seq=s, gen=0)).seq
            for s in (0, 1, 3)]
    missing = sorted(set(range(max(seqs) + 1)) - set(seqs))
    assert missing == [2]


# ── Control frames ──────────────────────────────────────────────────────────

def test_control_frames_carry_their_type():
    assert P.phase("listening")["value"] == "listening"
    assert P.generation(4)["n"] == 4
    assert P.dropped(2, "synthesis failed")["seq"] == 2
    assert P.turn_complete("hi", "hello", chunks=3)["chunks"] == 3
    assert P.engine_switch("realtime", "staged", "boom")["from"] == "realtime"
    assert P.error("x", "y", recoverable=False)["recoverable"] is False
    u = P.usage(10, 20, spent_usd=0.123456, ceiling_pct=0.5)
    assert u["input_tokens"] == 10 and u["spent_usd"] == 0.1235
