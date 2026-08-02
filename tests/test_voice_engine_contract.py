"""Engine contract + correctness properties (design §"Correctness Properties").

Each test here maps to a property whose violation is one of the five verified
defects. The point is not coverage — it is that the defects become
*unrepresentable* rather than merely fixed.

No network, no model, no audio device: synthesis is stubbed so ordering and
completion logic can be driven deterministically, including the failure modes
that are hard to reproduce on real hardware.
"""
from __future__ import annotations

import asyncio

import pytest

from app.voice import budget, policy, transcript
from app.voice.engine import (
    AudioDelta, AudioDropped, InterruptReason, Phase, PhaseChange,
    SessionLedger, SpeechInterrupted, TranscriptDelta, TurnComplete,
    UsageDelta, VoiceContext, VoiceTurn,
)
from app.voice.staged import StagedEngine


async def _drain(session, *, stop_on=TurnComplete, limit=200, trailing=2):
    """Collect events until `stop_on` is seen (or `limit`).

    `trailing` keeps reading a couple of events PAST the stop condition, because
    the return to `listening` is emitted immediately after `TurnComplete` and
    stopping dead on the completion would hide it.
    """
    out = []
    it = session.events()
    seen = False
    extra = 0
    for _ in range(limit):
        try:
            ev = await asyncio.wait_for(it.__anext__(), timeout=2.0)
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        out.append(ev)
        if seen:
            extra += 1
            if extra >= trailing:
                break
        elif isinstance(ev, stop_on):
            seen = True
            if trailing <= 0:
                break
    return out


@pytest.fixture()
def stub_tts(monkeypatch):
    """Deterministic synthesis with per-chunk control over latency + failure."""
    plan: dict = {"delays": {}, "fail": set(), "hang": set()}

    async def _synth(text, voice=None, *, speed=1.0):
        idx = int(text.split("-")[-1]) if "-" in text else 0
        if idx in plan["hang"]:
            await asyncio.sleep(60)
        await asyncio.sleep(plan["delays"].get(idx, 0.0))
        if idx in plan["fail"]:
            raise RuntimeError("synthesis backend exploded")
        return f"audio:{text}".encode()

    import app.live.tts_synth as tts
    monkeypatch.setattr(tts, "synthesize", _synth)
    return plan


async def _open():
    session = await StagedEngine().open(VoiceContext(conversation_id="c1"))
    return session


# ── Property 3: every turn terminates ───────────────────────────────────────

def test_turn_completes_exactly_once(stub_tts):
    """Regression for defects 2 and 3: a leaked hold used to keep the phase
    machine in `speaking` forever, and the client's 15s watchdog was the only
    way out. Completion is now server-side and unconditional."""
    async def _body():
        s = await _open()
        await s.send_text("what is kafka")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await s.reply_end(3)

        events = await _drain(s)
        completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(completes) == 1
        assert completes[0].chunks == 3
        # …and the phase machine returns to listening.
        assert any(isinstance(e, PhaseChange) and e.phase is Phase.LISTENING
                   for e in events)
        await s.close()
    asyncio.run(_body())


def test_a_failed_chunk_still_completes_the_turn(stub_tts):
    """Requirement 2.4 — an expected frame that never arrives must not leave the
    surface waiting. The engine reports the drop and finishes."""
    async def _body():
        stub_tts["fail"].add(1)
        s = await _open()
        await s.send_text("q")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await s.reply_end(3)

        events = await _drain(s)
        dropped = [e for e in events if isinstance(e, AudioDropped)]
        assert [d.seq for d in dropped] == [1]
        assert "synthesis failed" in dropped[0].reason
        assert len([e for e in events if isinstance(e, TurnComplete)]) == 1
        await s.close()
    asyncio.run(_body())


# ── Property 2: audio plays in emission order ──────────────────────────────

def test_a_slow_middle_chunk_cannot_be_overtaken(stub_tts):
    """Regression for defect 4 (mixed-transport ordering).

    Chunk 1 is slow — as it would be falling back from Kokoro to Edge — while 0
    and 2 are fast. Under the old single queue, 2 could be appended ahead of 1.
    The engine now holds 2 back until 1 has been emitted.
    """
    async def _body():
        stub_tts["delays"] = {0: 0.0, 1: 0.25, 2: 0.0}
        s = await _open()
        await s.send_text("q")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await s.reply_end(3)

        events = await _drain(s)
        audio = [e for e in events if isinstance(e, AudioDelta)]
        assert [a.seq for a in audio] == [0, 1, 2]
        assert [a.payload for a in audio] == [
            b"audio:chunk-0", b"audio:chunk-1", b"audio:chunk-2"]
        await s.close()
    asyncio.run(_body())


def test_order_holds_when_every_chunk_finishes_backwards(stub_tts):
    """The strongest form: synthesis completes in exactly reverse order."""
    async def _body():
        stub_tts["delays"] = {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.0}
        s = await _open()
        await s.send_text("q")
        for i in range(4):
            await s.speak(i, f"chunk-{i}")
        await s.reply_end(4)

        audio = [e for e in await _drain(s) if isinstance(e, AudioDelta)]
        assert [a.seq for a in audio] == [0, 1, 2, 3]
        await s.close()
    asyncio.run(_body())


# ── Property 5: interruption is atomic ─────────────────────────────────────

def test_interruption_drops_queued_audio_and_closes_the_turn(stub_tts):
    async def _body():
        stub_tts["delays"] = {0: 0.0, 1: 0.5, 2: 0.5}
        s = await _open()
        await s.send_text("q")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await asyncio.sleep(0.05)          # chunk 0 lands
        await s.interrupt(InterruptReason.TAP)

        events = await _drain(s)
        assert any(isinstance(e, SpeechInterrupted) for e in events)
        completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(completes) == 1 and completes[0].interrupted is True
        await s.close()
    asyncio.run(_body())


def test_audio_synthesized_after_a_barge_in_is_never_emitted(stub_tts):
    """Nothing from the interrupted reply reaches the speaker, even though its
    synthesis was already in flight when the interruption landed."""
    async def _body():
        stub_tts["delays"] = {0: 0.4}
        s = await _open()
        await s.send_text("q")
        await s.speak(0, "chunk-0")
        await asyncio.sleep(0.02)
        await s.interrupt(InterruptReason.SPEECH)
        await asyncio.sleep(0.6)           # the stale synthesis finishes now

        events = await _drain(s, stop_on=type(None), limit=40)
        assert not [e for e in events if isinstance(e, AudioDelta)]
        await s.close()
    asyncio.run(_body())


def test_generation_bump_is_monotonic_across_interrupts(stub_tts):
    async def _body():
        s = await _open()
        assert s.generation == 0
        await s.interrupt(InterruptReason.TAP)
        await s.interrupt(InterruptReason.SPEECH)
        assert s.generation == 2
        await s.close()
    asyncio.run(_body())


# ── First-real-word gate ────────────────────────────────────────────────────

def test_a_lone_filler_never_takes_a_turn(stub_tts):
    async def _body():
        s = await _open()
        await s.on_utterance("uh")
        await s.on_utterance("mm hmm")
        events = await _drain(s, stop_on=type(None), limit=8)
        assert not [e for e in events if isinstance(e, TranscriptDelta)]
        await s.close()
    asyncio.run(_body())


def test_a_real_short_reply_does_take_a_turn(stub_tts):
    async def _body():
        s = await _open()
        await s.on_utterance("yes")
        events = await _drain(s, stop_on=PhaseChange, limit=8)
        assert any(isinstance(e, TranscriptDelta) and e.text == "yes"
                   for e in events)
        await s.close()
    asyncio.run(_body())


# ── Preflight loads nothing (Requirement 6.4 / Property 9) ─────────────────

def test_staged_preflight_is_cheap_and_always_ok():
    pre = StagedEngine().preflight()
    assert pre.ok and pre.turn_detection == "server_vad"


# ── Property 8: cost is bounded ─────────────────────────────────────────────

def test_default_config_can_never_select_realtime():
    """Requirement 9.1 — default config means metered spend is exactly zero."""
    sel = policy.select()
    assert sel.engine == policy.STAGED


def test_realtime_is_refused_without_a_model_even_when_requested(monkeypatch):
    """The empty-model interlock outranks `engine: realtime`."""
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.voice, "engine", "realtime", raising=False)
    monkeypatch.setattr(cfg.voice, "realtime_model", "", raising=False)
    sel = policy.select()
    assert sel.engine == policy.STAGED
    assert "no realtime model" in sel.reason


def test_zero_ceiling_reads_as_no_budget_not_unlimited():
    """The safe reading of an unconfigured budget is that nothing may be spent.
    A ceiling of 0 must NOT mean 'unlimited'."""
    ok, why = budget.can_open_session()
    assert ok is False and "no voice budget" in why
    assert budget.remaining() == 0.0


def test_legacy_s2s_engine_key_still_maps(monkeypatch):
    """Requirement 6.5 — the superseded key is honoured for one release."""
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.voice, "engine", "staged", raising=False)
    monkeypatch.setattr(cfg.voice, "s2s_engine", "omni", raising=False)
    assert policy.configured_engine() == policy.REALTIME


# ── Property 7: handover preserves the conversation ────────────────────────

def test_ledger_history_survives_for_handover():
    ledger = SessionLedger()
    ledger.record(VoiceTurn("q1", "a1", engine="realtime"))
    ledger.record(VoiceTurn("q2", "a2", engine="realtime"))
    assert ledger.history() == (("q1", "a1"), ("q2", "a2"))


def test_metering_accumulates_on_the_ledger():
    ledger = SessionLedger()
    ledger.meter(UsageDelta(input_tokens=100, output_tokens=50))
    ledger.meter(UsageDelta(input_tokens=10, output_tokens=5))
    assert (ledger.input_tokens, ledger.output_tokens) == (110, 55)


# ── Requirement 10.3: turns neither lost nor duplicated ────────────────────

def test_staged_turns_are_not_re_persisted():
    """The client already wrote them through the normal chat path. Persisting
    again here would double every message in the thread."""
    ledger = SessionLedger()
    ledger.record(VoiceTurn("q", "a", engine="staged"))
    assert transcript.pending(ledger) == []


def test_realtime_turns_are_persisted():
    """Nothing wrote them — the server owned generation, so this module must."""
    ledger = SessionLedger()
    ledger.record(VoiceTurn("q", "a", engine="realtime"))
    assert len(transcript.pending(ledger)) == 1


def test_a_mixed_ledger_after_handover_persists_only_the_realtime_half():
    """One session can legitimately contain both kinds. Getting this wrong in
    either direction violates Requirement 10.3."""
    ledger = SessionLedger()
    ledger.record(VoiceTurn("q1", "a1", engine="realtime"))
    ledger.record(VoiceTurn("q2", "a2", engine="staged"))
    ledger.record(VoiceTurn("q3", "a3", engine="realtime"))
    assert [t.user for t in transcript.pending(ledger)] == ["q1", "q3"]


def test_empty_turns_are_not_persisted():
    ledger = SessionLedger()
    ledger.record(VoiceTurn("", "", engine="realtime"))
    assert transcript.pending(ledger) == []


# ── Interruption truncation (Property 5, the transcript half) ────────────────
#
# "Stops playback" is only half of an atomic barge-in. The other half is that
# the RECORDED answer matches what was heard: chunks that were queued but never
# spoken must not appear in the transcript, or the user reads words that were
# never said and the chat history is a record of a conversation that did not
# happen.

def test_an_interrupted_turn_records_only_the_spoken_chunks(stub_tts):
    async def _body():
        # Chunk 0 lands immediately; 1 and 2 are still synthesizing when the
        # barge-in arrives, so they were never heard.
        stub_tts["delays"] = {0: 0.0, 1: 0.6, 2: 0.6}
        s = await _open()
        await s.send_text("q")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await asyncio.sleep(0.08)          # chunk 0 emitted
        await s.interrupt(InterruptReason.TAP)

        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.interrupted is True
        assert done.assistant == "chunk-0", (
            f"recorded text the user never heard: {done.assistant!r}")
        await s.close()
    asyncio.run(_body())


def test_interrupting_before_any_audio_records_no_answer(stub_tts):
    """Nothing was spoken, so there is no answer to record — an empty string is
    the honest result, not the text we were about to say."""
    async def _body():
        stub_tts["delays"] = {0: 0.6}
        s = await _open()
        await s.send_text("q")
        await s.speak(0, "never heard")
        await s.interrupt(InterruptReason.SPEECH)
        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.assistant == ""
        await s.close()
    asyncio.run(_body())


def test_a_clean_turn_still_records_everything(stub_tts):
    """Truncation must fire ONLY on an interruption."""
    async def _body():
        s = await _open()
        await s.send_text("q")
        for i in range(3):
            await s.speak(i, f"chunk-{i}")
        await s.reply_end(3)
        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.interrupted is False
        assert done.assistant == "chunk-0 chunk-1 chunk-2"
        await s.close()
    asyncio.run(_body())


def test_the_next_turn_after_an_interruption_starts_clean(stub_tts):
    """Leftover chunk bookkeeping would contaminate the following answer."""
    async def _body():
        stub_tts["delays"] = {0: 0.0, 1: 0.6}
        s = await _open()
        await s.send_text("q1")
        await s.speak(0, "chunk-0")
        await s.speak(1, "chunk-1")
        await asyncio.sleep(0.08)
        await s.interrupt(InterruptReason.TAP)
        await _drain(s)

        stub_tts["delays"] = {}
        await s.send_text("q2")
        await s.speak(0, "fresh-0")
        await s.reply_end(1)
        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.user == "q2" and done.assistant == "fresh-0"
        await s.close()
    asyncio.run(_body())


def test_interrupt_accepts_played_ms_so_both_engines_share_a_signature():
    """The runner calls one method whichever engine is live."""
    import inspect

    from app.voice.realtime import RealtimeSession
    from app.voice.staged import StagedSession
    for cls in (StagedSession, RealtimeSession):
        params = list(inspect.signature(cls.interrupt).parameters)
        assert "played_ms" in params, f"{cls.__name__}.interrupt lacks played_ms"


def test_emitted_is_not_heard_the_client_report_wins(stub_tts):
    """The bug an end-to-end run found that these unit tests had missed.

    Earlier interruption tests used slow synthesis, so the unspoken chunks were
    still IN FLIGHT when the barge-in landed and cutting on the emit boundary
    happened to give the right answer. With fast synthesis the server pushes
    every chunk into the socket long before the user has heard the first one —
    and the emit boundary then records three sentences when one was spoken.

    Only the client knows where hearing stopped. This pins that its report is
    what decides, independently of how fast synthesis happened to be.
    """
    async def _body():
        stub_tts["delays"] = {}            # instant: ALL chunks emit at once
        s = await _open()
        await s.send_text("q")
        for i, part in enumerate(["First.", "Second.", "Third."]):
            await s.speak(i, part)
        await asyncio.sleep(0.1)           # everything is emitted by now
        await s.interrupt(InterruptReason.TAP, played_ms=100, played_chunks=1)

        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.assistant == "First.", (
            "the transcript followed what was EMITTED, not what was heard: "
            f"{done.assistant!r}")
    asyncio.run(_body())


def test_a_client_reporting_zero_playback_records_nothing(stub_tts):
    async def _body():
        s = await _open()
        await s.send_text("q")
        await s.speak(0, "never heard")
        await asyncio.sleep(0.05)
        await s.interrupt(InterruptReason.TAP, played_ms=0)
        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.assistant == ""
    asyncio.run(_body())


def test_a_client_that_reports_nothing_keeps_todays_behaviour(stub_tts):
    """An older client sends no playback figures at all. It must degrade to the
    emit boundary rather than losing the answer entirely."""
    async def _body():
        stub_tts["delays"] = {}
        s = await _open()
        await s.send_text("q")
        await s.speak(0, "spoken")
        await asyncio.sleep(0.08)
        await s.interrupt(InterruptReason.TAP)
        done = [e for e in await _drain(s) if isinstance(e, TurnComplete)][0]
        assert done.assistant == "spoken"
    asyncio.run(_body())
