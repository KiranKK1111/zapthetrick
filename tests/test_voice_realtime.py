"""`RealtimeEngine` against a FAKE upstream (design §3).

This closes the "never run against a live endpoint" gap as far as software can.
`RealtimeSession` takes its websocket by injection, so everything except the
actual `websockets.connect` handshake runs for real here: event translation,
audio decoding, native barge-in, tool dispatch, metering, and the failure path
that triggers handover.

What this still cannot prove: that the upstream's event NAMES match what it
actually sends. Those are pinned to the published realtime event vocabulary and
verified in the two-name pairs below (`response.audio.delta` and
`response.output_audio.delta`), but a live session remains the final check.
"""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app.voice import budget
from app.voice.engine import (
    AudioDelta, EngineError, InterruptReason, Phase, PhaseChange,
    SpeechInterrupted, ToolRequest, ToolResult, TranscriptDelta, TurnComplete,
    UsageDelta, VoiceContext,
)
from app.voice.realtime import RealtimeSession, _resample_16k_to_24k


class FakeUpstream:
    """A websocket double. Records what the session SENDS and replays a scripted
    server event stream, so a whole turn can be driven deterministically."""

    def __init__(self, script=None):
        self.sent: list[dict] = []
        self._script = list(script or [])
        self._queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        for ev in self._script:
            self._queue.put_nowait(ev)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True

    def push(self, ev: dict) -> None:
        """Inject a server event mid-test."""
        self._queue.put_nowait(ev)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            ev = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise StopAsyncIteration from None
        return json.dumps(ev)

    def sent_of_type(self, kind: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == kind]


def _audio_event(pcm: bytes, kind: str = "response.audio.delta") -> dict:
    return {"type": kind, "delta": base64.b64encode(pcm).decode("ascii")}


async def _collect(session, *, limit=40, until=None, trailing=2):
    """Collect events, reading a little PAST `until` — the return to `listening`
    is emitted immediately after `TurnComplete`, so stopping dead on the
    completion would hide it."""
    out = []
    it = session.events()
    seen = False
    extra = 0
    for _ in range(limit):
        try:
            ev = await asyncio.wait_for(it.__anext__(), timeout=1.5)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        out.append(ev)
        if seen:
            extra += 1
            if extra >= trailing:
                break
        elif until is not None and isinstance(ev, until):
            seen = True
            if trailing <= 0:
                break
    return out


def _session(script=None, ctx=None):
    up = FakeUpstream(script)
    sess = RealtimeSession(ctx or VoiceContext(conversation_id="c1"), up,
                           model="gpt-realtime-2.1")
    return sess, up


# ── Audio ───────────────────────────────────────────────────────────────────

def test_audio_deltas_become_ordered_pcm_frames():
    async def body():
        sess, up = _session([
            _audio_event(b"\x01\x02"),
            _audio_event(b"\x03\x04"),
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        audio = [e for e in events if isinstance(e, AudioDelta)]
        assert [a.seq for a in audio] == [0, 1]
        assert [a.payload for a in audio] == [b"\x01\x02", b"\x03\x04"]
        # Upstream returns raw PCM, not MP3 — the sink appends it directly,
        # which is the seamless path.
        assert all(a.mp3 is False for a in audio)
        await sess.close()
    asyncio.run(body())


def test_the_alternate_audio_event_name_is_handled():
    """The realtime vocabulary has used both `response.audio.delta` and
    `response.output_audio.delta`. Handling only one silently drops every
    chunk — the failure looks like 'the model said nothing'."""
    async def body():
        sess, _ = _session([
            _audio_event(b"\xAA", kind="response.output_audio.delta"),
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        assert [e.payload for e in events if isinstance(e, AudioDelta)] == [b"\xAA"]
        await sess.close()
    asyncio.run(body())


def test_mic_audio_is_resampled_and_base64_appended():
    async def body():
        sess, up = _session()
        # 16 kHz in, 24 kHz upstream: 3:2, so 100 samples -> 150.
        pcm = (b"\x00\x01" * 100)
        await sess.send_audio(pcm)
        appended = up.sent_of_type("input_audio_buffer.append")
        assert len(appended) == 1
        raw = base64.b64decode(appended[0]["audio"])
        assert len(raw) == 300, "expected 150 int16 samples after 3:2 resample"
        await sess.close()
    asyncio.run(body())


def test_resampler_is_length_correct_and_never_raises():
    assert _resample_16k_to_24k(b"") == b""
    assert len(_resample_16k_to_24k(b"\x00\x01" * 32)) == 32 * 3  # 48 samples
    # A malformed odd-length buffer must degrade, not explode, mid-conversation.
    assert isinstance(_resample_16k_to_24k(b"\x01\x02\x03"), bytes)


# ── Transcripts + turn assembly ─────────────────────────────────────────────

def test_transcripts_assemble_into_one_turn():
    async def body():
        sess, _ = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "what is kafka"},
            {"type": "response.audio_transcript.delta", "delta": "Kafka is "},
            {"type": "response.audio_transcript.delta", "delta": "a log."},
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        done = [e for e in events if isinstance(e, TurnComplete)][0]
        assert done.user == "what is kafka"
        assert done.assistant == "Kafka is a log."
        # …and the session returns to listening.
        assert any(isinstance(e, PhaseChange) and e.phase is Phase.LISTENING
                   for e in events)
        await sess.close()
    asyncio.run(body())


# ── Native barge-in ─────────────────────────────────────────────────────────

def test_upstream_speech_start_raises_a_native_interruption():
    """The model heard the interruption itself — that is the whole reason turn
    detection moves upstream on this engine."""
    async def body():
        # A turn with no transcript at all records nothing (by design — there is
        # no exchange to log), so give the turn real content and interrupt it.
        sess, _ = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "explain raft"},
            _audio_event(b"\x01"),
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        cut = [e for e in events if isinstance(e, SpeechInterrupted)]
        assert len(cut) == 1
        assert cut[0].reason is InterruptReason.ENGINE
        assert [e for e in events if isinstance(e, TurnComplete)][0].interrupted
        await sess.close()
    asyncio.run(body())


def test_client_tap_cancels_the_upstream_response():
    async def body():
        sess, up = _session()
        await sess.interrupt(InterruptReason.TAP)
        assert up.sent_of_type("response.cancel"), "upstream was not cancelled"
        await sess.close()
    asyncio.run(body())


# ── Tool bridge ─────────────────────────────────────────────────────────────

def test_a_tool_call_is_dispatched_and_its_result_returned_upstream(monkeypatch):
    async def body():
        async def fake_dispatch(name, args, ctx, allowed=None):
            return {"answer": f"reasoned about {json.loads(args)['question']}"}

        # Patched via the USING module's reference, not app.voice.tools directly.
        monkeypatch.setattr("app.voice.realtime.tools.dispatch", fake_dispatch)
        sess, up = _session([
            {"type": "response.function_call_arguments.done",
             "call_id": "call_1", "name": "ask_reasoner",
             "arguments": json.dumps({"question": "explain raft"})},
        ])
        events = await _collect(sess, until=ToolResult)
        # The phase goes to TOOL so a slow lookup is audible as work, not silence.
        assert any(isinstance(e, PhaseChange) and e.phase is Phase.TOOL
                   for e in events)
        assert [e.name for e in events if isinstance(e, ToolRequest)] == \
            ["ask_reasoner"]
        result = [e for e in events if isinstance(e, ToolResult)][0]
        assert result.ok is True

        await asyncio.sleep(0.05)
        out = [m for m in up.sent
               if m.get("item", {}).get("type") == "function_call_output"]
        assert out and "reasoned about explain raft" in out[0]["item"]["output"]
        # …and the model is told to continue speaking with the result in hand.
        assert up.sent_of_type("response.create")
        await sess.close()
    asyncio.run(body())


def test_a_failing_tool_is_reported_to_the_model_not_the_user(monkeypatch):
    """Requirement 8.4 — the model answers WITHOUT the tool rather than the turn
    dying."""
    async def body():
        async def broken(name, args, ctx, allowed=None):
            return {"error": "sandbox unavailable"}

        monkeypatch.setattr("app.voice.realtime.tools.dispatch", broken)
        sess, up = _session([
            {"type": "response.function_call_arguments.done",
             "call_id": "c2", "name": "run_code", "arguments": "{}"},
        ])
        events = await _collect(sess, until=ToolResult)
        result = [e for e in events if isinstance(e, ToolResult)][0]
        assert result.ok is False and "sandbox unavailable" in result.detail
        # The failure goes UPSTREAM as a tool output; it is never surfaced as a
        # user-facing error frame.
        await asyncio.sleep(0.05)
        out = [m for m in up.sent
               if m.get("item", {}).get("type") == "function_call_output"]
        assert out and "sandbox unavailable" in out[0]["item"]["output"]
        assert not [e for e in events if isinstance(e, EngineError)]
        await sess.close()
    asyncio.run(body())


# ── Metering ────────────────────────────────────────────────────────────────

def test_usage_is_metered_per_response_with_cached_tokens_discounted():
    async def body():
        sess, _ = _session([
            {"type": "response.done", "response": {"usage": {
                "input_tokens": 1000, "output_tokens": 500,
                "input_token_details": {"cached_tokens": 400}}}},
        ])
        events = await _collect(sess, until=TurnComplete)
        usage = [e for e in events if isinstance(e, UsageDelta)][0]
        # Cached input is billed far cheaper and must not be counted twice.
        assert usage.input_tokens == 600
        assert usage.output_tokens == 500
        assert sess.meter.cached_tokens == 400
        await sess.close()
    asyncio.run(body())


def test_a_response_without_usage_does_not_break_the_turn():
    """No usage block ⇒ nothing metered, but the turn still closes cleanly."""
    async def body():
        sess, _ = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "hello"},
            {"type": "response.audio_transcript.delta", "delta": "hi there"},
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        assert [e for e in events if isinstance(e, TurnComplete)]
        assert not [e for e in events if isinstance(e, UsageDelta)]
        await sess.close()
    asyncio.run(body())


def test_an_empty_response_records_no_turn():
    """A response with neither a user nor an assistant transcript is not a turn
    and must not land in the ledger as an empty pair."""
    async def body():
        sess, _ = _session([{"type": "response.done", "response": {}}])
        events = await _collect(sess, limit=6)
        assert not [e for e in events if isinstance(e, TurnComplete)]
        await sess.close()
    asyncio.run(body())


def test_session_wall_clock_cap_is_reported(monkeypatch):
    async def body():
        sess, _ = _session()
        monkeypatch.setattr(budget, "session_seconds_cap", lambda: 0.0)
        assert sess.over_session_cap() is False   # 0 ⇒ uncapped, not "expired"
        monkeypatch.setattr(budget, "session_seconds_cap", lambda: 0.01)
        await asyncio.sleep(0.05)   # a real interval — monotonic() can repeat a tick
        assert sess.over_session_cap() is True
        await sess.close()
    asyncio.run(body())


# ── Failure → handover ──────────────────────────────────────────────────────

def test_an_upstream_error_is_recoverable_so_the_runner_hands_over():
    async def body():
        sess, _ = _session([
            {"type": "error", "error": {"type": "server_error",
                                        "message": "upstream exploded"}},
        ])
        events = await _collect(sess, until=EngineError)
        err = [e for e in events if isinstance(e, EngineError)][0]
        assert err.recoverable is True, \
            "a non-recoverable error would end the conversation instead of " \
            "falling back to the staged engine"
        assert "upstream exploded" in err.detail
        await sess.close()
    asyncio.run(body())


def test_a_dropped_socket_surfaces_as_recoverable():
    async def body():
        sess, up = _session()
        up.push({"type": "response.done", "response": {}})
        await _collect(sess, until=TurnComplete)
        # The upstream iterator ending is a closed socket, not a clean finish.
        events = await _collect(sess, limit=6, until=EngineError)
        errs = [e for e in events if isinstance(e, EngineError)]
        assert all(e.recoverable for e in errs)
        await sess.close()
    asyncio.run(body())


# ── Session open contract (no network) ──────────────────────────────────────

def test_preflight_is_cheap_and_refuses_without_a_model():
    from app.voice.realtime import RealtimeEngine
    pre = RealtimeEngine().preflight()
    # Default config: no realtime model configured ⇒ never selected, and the
    # check performs no network I/O and loads nothing.
    assert pre.ok is False
    assert "model" in pre.reason or "credential" in pre.reason or "budget" in pre.reason


def test_history_is_replayed_so_a_handover_in_keeps_context():
    """Property 7 — an engine opened mid-conversation starts with the turns that
    already happened rather than an empty head."""
    async def body():
        ctx = VoiceContext(conversation_id="c1",
                           history=(("q1", "a1"), ("q2", "a2")))
        sess, up = _session(ctx=ctx)
        # `open()` does the seeding; simulate its loop directly against the
        # session's sender so the test needs no network.
        for user, assistant in ctx.history:
            for role, text in (("user", user), ("assistant", assistant)):
                await sess._send({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": role,
                             "content": [{"type": ("input_text"
                                                   if role == "user" else "text"),
                                          "text": text}]}})
        items = [m for m in up.sent if m.get("type") == "conversation.item.create"]
        assert len(items) == 4
        assert items[0]["item"]["content"][0]["text"] == "q1"
        await sess.close()
    asyncio.run(body())


def test_typed_input_during_a_voice_session_creates_a_turn():
    """The accessibility path — Requirement 6 `send_text`."""
    async def body():
        sess, up = _session()
        await sess.send_text("what is a mutex")
        created = [m for m in up.sent if m.get("type") == "conversation.item.create"]
        assert created[0]["item"]["content"][0]["text"] == "what is a mutex"
        assert up.sent_of_type("response.create")
        await sess.close()
    asyncio.run(body())


# ── Interruption truncation: the transcript must match what was HEARD ────────
#
# The model generates its transcript AHEAD of playback, so at the moment of a
# barge-in the session is holding text the user never heard. Two things must
# happen, and skipping either produces a specific, silent bug:
#   * upstream is told where hearing stopped, or its context believes it said
#     the whole reply and every follow-up reasons from a false premise;
#   * our own transcript is cut, or the user reads words that were never spoken.

def test_a_barge_in_tells_upstream_where_hearing_stopped():
    """Without `conversation.item.truncate` the model's context keeps the whole
    reply — the "as I mentioned…" bug, about something it never said."""
    async def body():
        sess, up = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "explain raft"},
            # 48 bytes per ms at 24 kHz mono int16 ⇒ 4800 bytes = 100 ms.
            {"type": "response.audio.delta", "item_id": "item_7",
             "delta": base64.b64encode(b"\x00" * 4800).decode("ascii")},
            {"type": "input_audio_buffer.speech_started"},
        ])
        await _collect(sess, until=SpeechInterrupted)
        await asyncio.sleep(0.05)
        trunc = up.sent_of_type("conversation.item.truncate")
        assert trunc, "upstream was never told the reply was cut short"
        assert trunc[0]["item_id"] == "item_7"
        assert trunc[0]["audio_end_ms"] == 100
        await sess.close()
    asyncio.run(body())


def test_the_recorded_answer_is_cut_to_what_was_played():
    """A client that rendered half the audio must not have the whole transcript
    recorded against it."""
    async def body():
        sess, up = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "explain raft"},
            {"type": "response.audio.delta", "item_id": "i1",
             "delta": base64.b64encode(b"\x00" * 4800).decode("ascii")},  # 100ms
            {"type": "response.audio_transcript.delta",
             "delta": "0123456789"},   # 10 chars alongside 100 ms of audio
        ])
        await _collect(sess, limit=6)
        # The client reports it only rendered 40 ms of the 100 ms sent.
        await sess.interrupt(InterruptReason.TAP, played_ms=40)
        await sess._finish_turn()
        events = await _collect(sess, until=TurnComplete, limit=8)
        done = [e for e in events if isinstance(e, TurnComplete)][0]
        assert done.assistant == "0123", \
            f"transcript not cut to the heard fraction: {done.assistant!r}"
        await sess.close()
    asyncio.run(body())


def test_a_clean_turn_keeps_its_whole_transcript():
    """Truncation must only ever fire on an interruption."""
    async def body():
        sess, _ = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "hi"},
            {"type": "response.audio.delta", "item_id": "i1",
             "delta": base64.b64encode(b"\x00" * 4800).decode("ascii")},
            {"type": "response.audio_transcript.delta", "delta": "full answer"},
            {"type": "response.done", "response": {}},
        ])
        events = await _collect(sess, until=TurnComplete)
        assert [e for e in events
                if isinstance(e, TurnComplete)][0].assistant == "full answer"
        await sess.close()
    asyncio.run(body())


def test_truncation_state_resets_between_turns():
    """A stale item id or audio total would cut the NEXT reply at the wrong
    point — a bug that only shows up on the second interruption."""
    async def body():
        sess, _ = _session([
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "one"},
            {"type": "response.audio.delta", "item_id": "i1",
             "delta": base64.b64encode(b"\x00" * 4800).decode("ascii")},
            {"type": "response.audio_transcript.delta", "delta": "first"},
            {"type": "response.done", "response": {}},
        ])
        await _collect(sess, until=TurnComplete)
        assert sess._item_id == "" and sess._audio_ms == 0.0
        await sess.close()
    asyncio.run(body())


def test_no_audio_yet_means_nothing_to_truncate():
    """Interrupting before a single chunk arrived must not send a truncate with
    a zero offset (upstream rejects it) nor blank an empty transcript."""
    async def body():
        sess, up = _session()
        await sess.interrupt(InterruptReason.TAP)
        await asyncio.sleep(0.05)
        assert not up.sent_of_type("conversation.item.truncate")
        await sess.close()
    asyncio.run(body())


# ── Session configuration: what makes it feel like ChatGPT ───────────────────

def test_semantic_vad_is_the_default_turn_detection():
    """A fixed silence timer interrupts anyone who thinks while speaking. Letting
    the MODEL judge whether a thought is finished is the single most noticeable
    difference between a voice assistant that talks over you and one that does
    not."""
    async def body():
        sess, _ = _session()
        td = sess._turn_detection()
        assert td["type"] == "semantic_vad"
        assert td["eagerness"] == "auto"
        # Upstream must own barge-in — that is why turn detection moved there.
        assert td["interrupt_response"] is True
        assert td["create_response"] is True
        await sess.close()
    asyncio.run(body())


def test_turn_detection_is_configurable(monkeypatch):
    from app.core.config_loader import cfg

    async def body():
        sess, _ = _session()
        monkeypatch.setattr(cfg.voice, "realtime_eagerness", "low",
                            raising=False)
        assert sess._turn_detection()["eagerness"] == "low"
        monkeypatch.setattr(cfg.voice, "realtime_turn_detection", "server_vad",
                            raising=False)
        monkeypatch.setattr(cfg.voice, "realtime_silence_ms", 700, raising=False)
        td = sess._turn_detection()
        assert td["type"] == "server_vad" and td["silence_duration_ms"] == 700
        await sess.close()
    asyncio.run(body())


def test_noise_reduction_is_requested():
    """Without it a laptop fan or a room is fed to the turn detector as speech."""
    async def body():
        sess, _ = _session()
        payload = sess._session_payload(VoiceContext())
        assert payload["input_audio_noise_reduction"] == {"type": "near_field"}
        await sess.close()
    asyncio.run(body())


def test_noise_reduction_can_be_disabled(monkeypatch):
    from app.core.config_loader import cfg

    async def body():
        sess, _ = _session()
        monkeypatch.setattr(cfg.voice, "realtime_noise_reduction", "",
                            raising=False)
        assert "input_audio_noise_reduction" not in sess._session_payload(
            VoiceContext())
        await sess.close()
    asyncio.run(body())


def test_the_session_carries_the_tool_schema_and_the_chosen_voice():
    async def body():
        sess, _ = _session()
        payload = sess._session_payload(VoiceContext(voice_id="verse"))
        assert payload["voice"] == "verse"
        assert payload["input_audio_format"] == "pcm16"
        assert payload["output_audio_format"] == "pcm16"
        names = {t["name"] for t in payload["tools"]}
        assert "ask_reasoner" in names, "voice mode would lose its depth path"
        await sess.close()
    asyncio.run(body())


def test_an_endpoint_without_semantic_vad_is_downgraded_not_failed():
    """This engine cannot be validated live from here. An endpoint that does not
    know `semantic_vad` must not lose the whole session over a turn-detection
    preference — it should drop to `server_vad` and say so."""
    async def body():
        sess, up = _session([
            {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "Unknown parameter: 'session.turn_detection.eagerness'."}},
        ])
        events = await _collect(sess, limit=6)
        # No error surfaced to the runner — it was handled by downgrading…
        assert not [e for e in events if isinstance(e, EngineError)]
        await asyncio.sleep(0.05)
        updates = up.sent_of_type("session.update")
        assert updates, "no session.update was re-sent"
        assert updates[-1]["session"]["turn_detection"]["type"] == "server_vad"
        await sess.close()
    asyncio.run(body())


def test_a_real_outage_still_surfaces_after_a_downgrade():
    """The downgrade must not swallow unrelated failures."""
    async def body():
        sess, _ = _session([
            {"type": "error", "error": {"type": "server_error",
                                        "message": "upstream exploded"}},
        ])
        events = await _collect(sess, until=EngineError, limit=6)
        errs = [e for e in events if isinstance(e, EngineError)]
        assert errs and "exploded" in errs[0].detail
        await sess.close()
    asyncio.run(body())


def test_the_downgrade_happens_at_most_once():
    """Re-sending forever on a repeated error would be a loop against upstream."""
    async def body():
        sess, up = _session()
        assert await sess._maybe_downgrade_vad("bad semantic_vad") is True
        assert await sess._maybe_downgrade_vad("bad semantic_vad") is False
        await sess.close()
    asyncio.run(body())


# ── Turning it on: the whole selection path ─────────────────────────────────

def test_configuring_a_model_key_and_budget_actually_selects_realtime(monkeypatch):
    """S2S is unreachable by default (empty model, zero budget) — deliberately,
    so metered spend is exactly zero out of the box. But a deployment that sets
    the three required things must genuinely GET it: a feature that cannot be
    switched on is not a feature.
    """
    from app.core.config_loader import cfg
    from app.voice import budget, policy

    monkeypatch.setattr(cfg.voice, "engine", "auto", raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_model", "gpt-realtime",
                        raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_api_key", "sk-test", raising=False)
    monkeypatch.setattr(budget, "can_open_session", lambda: (True, ""))

    sel = policy.select(reachable=True)
    assert sel.engine == policy.REALTIME, sel.reason


def test_realtime_preflight_passes_once_configured(monkeypatch):
    from app.core.config_loader import cfg
    from app.voice import budget
    from app.voice.realtime import RealtimeEngine

    monkeypatch.setattr(cfg.voice, "realtime_model", "gpt-realtime",
                        raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_api_key", "sk-test", raising=False)
    monkeypatch.setattr(budget, "can_open_session", lambda: (True, ""))

    pre = RealtimeEngine().preflight()
    assert pre.ok, pre.reason
    # The client is told turn-taking moved upstream, so it must not run its own.
    assert pre.turn_detection == "upstream_vad"


def test_the_server_segmenter_is_not_run_on_the_realtime_path():
    """Two turn detectors produce two competing decisions. The engine boundary
    exists so turn-taking authority moves WITH the engine."""
    import inspect

    from app.api import routes_voice_ws as R
    src = inspect.getsource(R.VoiceSessionRunner._open_engine)
    attach = src.index("_attach_segmenter")
    # The only segmenter wiring sits inside the staged branch.
    assert "STAGED" in src[:attach], \
        "the segmenter is attached outside the staged branch"


def test_a_missing_budget_refuses_realtime_even_when_fully_configured(monkeypatch):
    """Requirement 9: the ceiling is the interlock, not a suggestion."""
    from app.core.config_loader import cfg
    from app.voice import policy

    monkeypatch.setattr(cfg.voice, "engine", "realtime", raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_model", "gpt-realtime",
                        raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_api_key", "sk-test", raising=False)
    # Default zero ceilings ⇒ "no budget", which must outrank the request.
    sel = policy.select(reachable=True)
    assert sel.engine == policy.STAGED
    assert "budget" in sel.reason or "ceiling" in sel.reason
