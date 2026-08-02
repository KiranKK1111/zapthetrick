"""Chat VOICE MODE WebSocket — `/ws/voice`.

**Live does not live here.** This file exists because `/ws/voice` and `/ws/live`
used to be two functions in `app/api/routes_ws.py`, which meant every change to
voice mode edited the file the Live interview module lives in. Ending that
adjacency is the first structural requirement of the realtime-voice-mode design
(§L1), not a cleanup: it is what makes voice work incapable of regressing Live.

The runner is deliberately thin — it owns the socket, selects an engine,
forwards frames, and performs handover. No conversational logic lives here, so
the engines stay independently testable.

Wire protocol
-------------
v2 (``app/voice/protocol.py``): binary frames are self-describing, carrying
their own ``seq`` and ``gen``, so stale audio in flight during a barge-in is
dropped *by value* rather than by client-side bookkeeping. Control frames are
JSON. The client sees the same vocabulary whichever engine is running.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.voice import budget, policy, protocol, transcript
from app.voice.engine import (
    AudioDelta, AudioDropped, EngineError, InterruptReason, Phase, PhaseChange,
    SessionLedger, SpeechInterrupted, SttStatus, ToolRequest, ToolResult,
    TranscriptDelta,
    TurnComplete, UsageDelta, VoiceContext, VoiceTurn,
)
from app.voice.staged import StagedEngine

log = logging.getLogger("zapthetrick.voice.ws")

router = APIRouter()

# How often the runner checks the session wall-clock / budget ceiling.
_GOVERNOR_TICK_S = 5.0


def _engine_for(name: str):
    """Resolve an engine by name. Realtime is imported lazily so a build with no
    `websockets` extra, or no credential, never pays for the import."""
    if name == policy.REALTIME:
        from app.voice.realtime import RealtimeEngine
        return RealtimeEngine()
    return StagedEngine()


@router.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    voice: str | None = Query(default=None),
    speed: float = Query(default=1.0),
    language: str | None = Query(default=None),
):
    """Full-duplex spoken conversation.

    Up:   binary int16 PCM mono @16 kHz mic frames; JSON control frames.
    Down: protocol-v2 binary audio frames; JSON control frames.
    """
    from app.api.auth import authenticate_ws, ws_user_id
    from storage.context import current_user_id_var

    _uid, _err = authenticate_ws(token)
    if _err is not None:
        await websocket.close(code=1008)  # policy violation
        return
    # Bind the DEVICE user in auth-off mode (not None) so WS-path scoping
    # matches HTTP + resolve_user_id() — otherwise router/cache/catalog run
    # under NULL.
    _bound = await ws_user_id(_uid)
    if _bound:
        current_user_id_var.set(_bound)
    await websocket.accept()

    runner = VoiceSessionRunner(
        websocket,
        ctx=VoiceContext(
            user_id=str(_bound or _uid or "") or None,
            conversation_id=conversation_id,
            session_id=session_id,
            voice_id=(voice or ""),
            language=(language or "en"),
            speed=float(speed or 1.0),
        ),
    )
    await runner.run()


class VoiceSessionRunner:
    """Owns one voice socket: engine selection, frame forwarding, handover."""

    def __init__(self, websocket: WebSocket, *, ctx: VoiceContext) -> None:
        self._ws = websocket
        self._ctx = ctx
        self._ledger = SessionLedger()
        self._send_lock = asyncio.Lock()
        self._gen = 0                 # generation floor for stale-audio drops
        self._engine_name = policy.STAGED
        self._session = None
        self._segmenter = None
        self._pump: asyncio.Task | None = None
        self._governor: asyncio.Task | None = None
        self._closing = False
        self._meter = budget.SessionMeter()

    # ── frame I/O ───────────────────────────────────────────────────────────

    async def _send_json(self, obj: dict) -> None:
        try:
            async with self._send_lock:
                await self._ws.send_json(obj)
        except Exception:  # noqa: BLE001 — socket closing
            pass

    async def _send_audio(self, seq: int, payload: bytes, *, mp3: bool) -> None:
        """One self-describing binary frame. No paired meta frame: `seq` and
        `gen` travel INSIDE the frame, which is what makes a stale chunk
        droppable by value."""
        kind = (protocol.FrameKind.AUDIO_MP3 if mp3
                else protocol.FrameKind.AUDIO_PCM)
        try:
            frame = protocol.encode_audio(payload, kind=kind, seq=seq,
                                          gen=self._gen)
            async with self._send_lock:
                await self._ws.send_bytes(frame)
        except Exception:  # noqa: BLE001
            pass

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            await self._open_engine(initial=True)
            await self._read_client()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.info("voice session ended on error", exc_info=True)
        finally:
            await self._teardown()

    async def _open_engine(self, *, initial: bool) -> None:
        """Select and open an engine. Falls back to staged on any open failure —
        a user should never see an error because a cloud endpoint was slow."""
        selection = policy.select()
        name = selection.engine
        if name == policy.REALTIME:
            try:
                engine = _engine_for(name)
                pre = engine.preflight()
                if not pre.ok:
                    raise RuntimeError(pre.reason)
                self._session = await engine.open(self._ctx)
                self._engine_name = name
                turn_detection = pre.turn_detection
            except Exception as exc:  # noqa: BLE001
                log.info("voice: realtime unavailable (%s) — using staged", exc)
                selection = policy.fallback(f"realtime open failed: {exc}")
                name = policy.STAGED
                self._session = None
        if self._session is None:
            engine = _engine_for(policy.STAGED)
            self._session = await engine.open(self._ctx)
            self._engine_name = policy.STAGED
            turn_detection = engine.preflight().turn_detection
            await self._attach_segmenter()

        if not initial:
            await self._send_json(protocol.engine_switch(
                self._engine_name, name, selection.reason))

        await self._send_json(protocol.session_ready(
            self._engine_name, self._ctx.voice_id or "",
            turn_detection=turn_detection))
        await self._send_json(protocol.generation(self._gen))
        await self._send_json(protocol.phase(Phase.LISTENING.value))

        self._pump = asyncio.create_task(self._pump_events())
        if self._engine_name == policy.REALTIME:
            self._governor = asyncio.create_task(self._govern())

    async def _attach_segmenter(self) -> None:
        """Wire the SHARED Silero segmenter to the staged session.

        `app/audio/stream.py` is imported read-only and used with its existing
        signature — rule L2. The realtime path never builds one, because the
        upstream model owns turn detection there.
        """
        from app.audio.stream import AudioStreamSegmenter
        sess = self._session
        self._segmenter = AudioStreamSegmenter(
            on_utterance=sess.on_utterance,
            on_partial=sess.on_partial,
            prompt_provider=sess.bias_prompt,
            on_stt_status=sess.on_stt_status,
            on_speech_start=self._on_speech_start,
        )
        sess.attach_segmenter(self._segmenter)

    async def _on_speech_start(self) -> None:
        """Pre-open the answer provider's connection WHILE the user is still
        talking, so the reply's first request never pays TLS setup."""
        with contextlib.suppress(Exception):
            from app.perceived.prefetch import warm_live_provider
            await warm_live_provider()

    # ── engine → client ─────────────────────────────────────────────────────

    async def _pump_events(self) -> None:
        """Translate the engine's single ordered event stream into wire frames.

        Ordering is the ENGINE's guarantee, so this loop is a pure translation —
        it must never reorder, buffer or drop, or the correctness properties
        move back into the client where they were broken before.
        """
        try:
            async for ev in self._session.events():
                await self._forward(ev)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.info("voice event pump stopped", exc_info=True)

    async def _forward(self, ev) -> None:  # noqa: C901 — flat translation table
        if isinstance(ev, AudioDelta):
            await self._send_audio(ev.seq, ev.payload, mp3=ev.mp3)
            return
        if isinstance(ev, AudioDropped):
            # Property 1: abandoned audio is REPORTED, never silently lost.
            await self._send_json(protocol.dropped(ev.seq, ev.reason))
            return
        if isinstance(ev, TranscriptDelta):
            await self._send_json(protocol.transcript(
                ev.role, ev.text, final=ev.final, complete=ev.complete))
            return
        if isinstance(ev, SttStatus):
            # Informational, not an error: the session continues, but a dead
            # mic is explainable rather than silent.
            await self._send_json(protocol.stt_status(ev.state, ev.detail))
            return
        if isinstance(ev, PhaseChange):
            await self._send_json(protocol.phase(ev.phase.value))
            return
        if isinstance(ev, ToolRequest):
            await self._send_json(protocol.tool(ev.id, ev.name, "start"))
            return
        if isinstance(ev, ToolResult):
            await self._send_json(protocol.tool(
                ev.id, ev.name, "ok" if ev.ok else "fail"))
            return
        if isinstance(ev, SpeechInterrupted):
            # Bump the floor FIRST so anything still in flight is already stale
            # by the time the client sees the new generation.
            self._gen += 1
            await self._send_json(protocol.generation(self._gen))
            return
        if isinstance(ev, TurnComplete):
            self._ledger.record(VoiceTurn(
                user=ev.user, assistant=ev.assistant, engine=self._engine_name,
                interrupted=ev.interrupted))
            await self._send_json(protocol.turn_complete(
                ev.user, ev.assistant, interrupted=ev.interrupted,
                chunks=ev.chunks))
            return
        if isinstance(ev, UsageDelta):
            self._ledger.meter(ev)
            self._meter.add(input_tokens=ev.input_tokens,
                            output_tokens=ev.output_tokens)
            await self._send_json(protocol.usage(
                ev.input_tokens, ev.output_tokens,
                spent_usd=self._meter.usd,
                ceiling_pct=budget.ceiling_fraction()))
            if budget.should_warn():
                await self._send_json(protocol.error(
                    "budget_warning",
                    "Voice spend is approaching its ceiling.",
                    recoverable=True))
            return
        if isinstance(ev, EngineError):
            if ev.recoverable and self._engine_name == policy.REALTIME:
                await self._handover(ev.detail)
            else:
                await self._send_json(protocol.error(
                    ev.kind, ev.detail, recoverable=ev.recoverable))
            return

    # ── handover ────────────────────────────────────────────────────────────

    async def _handover(self, reason: str) -> None:
        """Realtime → staged without dropping the conversation (Property 7).

        The ledger and `conversation_id` are preserved, so completed turns are
        neither lost nor duplicated and the user keeps talking to the same
        thread.
        """
        if self._closing or self._engine_name != policy.REALTIME:
            return
        log.info("voice: handing over realtime → staged (%s)", reason)
        self._ledger.engine_switches.append(
            (policy.REALTIME, policy.STAGED, reason))
        await self._send_json(protocol.engine_switch(
            policy.REALTIME, policy.STAGED, reason))

        if self._governor:
            self._governor.cancel()
            self._governor = None
        if self._pump:
            self._pump.cancel()
            self._pump = None
        old, self._session = self._session, None
        with contextlib.suppress(Exception):
            await old.close()

        # Carry the history in, so the new engine starts where the old stopped.
        self._ctx.history = self._ledger.history()
        engine = _engine_for(policy.STAGED)
        self._session = await engine.open(self._ctx)
        self._engine_name = policy.STAGED
        await self._attach_segmenter()
        self._pump = asyncio.create_task(self._pump_events())
        await self._send_json(protocol.phase(Phase.LISTENING.value))

    async def _govern(self) -> None:
        """Stop a metered session at its wall-clock cap or spend ceiling.

        The CURRENT turn is allowed to finish (Requirement 9.4) — this only
        prevents the next one, by handing over.
        """
        try:
            while not self._closing and self._engine_name == policy.REALTIME:
                await asyncio.sleep(_GOVERNOR_TICK_S)
                sess = self._session
                if sess is None:
                    return
                if getattr(sess, "over_session_cap", lambda: False)():
                    await self._handover("session time cap reached")
                    return
                if self._meter.exhausted():
                    await self._handover("voice spend ceiling reached")
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    # ── client → engine ─────────────────────────────────────────────────────

    async def _read_client(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            raw = msg.get("bytes")
            if raw is not None:
                if self._session is not None:
                    await self._session.send_audio(raw)
                continue
            txt = msg.get("text")
            if not txt:
                continue
            try:
                ctrl = json.loads(txt)
            except Exception:  # noqa: BLE001
                continue
            if await self._control(ctrl):
                return


    @staticmethod
    def _played(ctrl: dict) -> dict:
        """What the client says it actually rendered.

        Two figures, because the two engines have different natural boundaries:
        `played_chunks` is exact for the staged engine (its transcript is
        assembled per chunk), and `played_ms` is what the realtime engine needs
        (its audio is one continuous stream). Both are forwarded; each engine
        uses the one it can act on precisely.
        """
        out: dict = {}
        ms = ctrl.get("played_ms")
        if isinstance(ms, (int, float)):
            out["played_ms"] = float(ms)
        chunks = ctrl.get("played_chunks")
        if isinstance(chunks, int):
            out["played_chunks"] = chunks
        return out

    async def _control(self, ctrl: dict) -> bool:
        """Handle one JSON control frame. Returns True to close the session."""
        kind = str(ctrl.get("type") or "")
        sess = self._session

        if kind in ("stop", "close"):
            return True

        if kind == "text":
            if sess is not None:
                await sess.send_text(str(ctrl.get("content") or ""))
            return False

        if kind == "interrupt":
            reason = (InterruptReason.TAP
                      if str(ctrl.get("reason") or "") == "tap"
                      else InterruptReason.SPEECH)
            # How much reply audio the client actually RENDERED. Only it knows —
            # the server has forwarded more than was heard whenever anything sat
            # in the jitter buffer — and it is what lets the transcript be cut
            # where the user stopped hearing rather than where we stopped sending.
            if sess is not None:
                await sess.interrupt(reason, **self._played(ctrl))
            return False

        # ── staged-only frames ─────────────────────────────────────────────
        # Under realtime the model generates its own reply, so these are inert
        # rather than errors: a client that sends them is simply ahead of the
        # engine switch it has not processed yet.
        if kind == "assistant_speaking":
            if hasattr(sess, "set_speaking"):
                sess.set_speaking(bool(ctrl.get("value")))
            return False

        if kind == "speak":
            if hasattr(sess, "speak"):
                await sess.speak(int(ctrl.get("seq") or 0),
                                 str(ctrl.get("text") or ""),
                                 voice=str(ctrl.get("voice") or ""),
                                 speed=float(ctrl.get("speed") or 1.0))
            return False

        if kind == "reply_end":
            if hasattr(sess, "reply_end"):
                total = ctrl.get("chunks")
                await sess.reply_end(int(total) if total is not None else None)
            return False

        if kind == "stop_speaking":
            if sess is not None:
                await sess.interrupt(InterruptReason.TAP, **self._played(ctrl))
            return False

        if kind == "flush":
            if self._segmenter is not None:
                with contextlib.suppress(Exception):
                    await self._segmenter.flush()
            return False

        return False

    # ── teardown ────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        self._closing = True
        for task in (self._governor, self._pump):
            if task is not None:
                task.cancel()
        if self._segmenter is not None:
            with contextlib.suppress(Exception):
                await self._segmenter.flush()
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
        # Spoken turns land in the chat thread. Only server-generated ones —
        # the staged path's turns were already persisted by the client.
        with contextlib.suppress(Exception):
            await transcript.persist_ledger(self._ledger,
                                            self._ctx.conversation_id)


__all__ = ["router", "voice_ws", "VoiceSessionRunner"]
