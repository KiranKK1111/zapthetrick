"""`RealtimeEngine` — a speech-native conversation session (design §3).

This is the engine that makes voice mode feel like ChatGPT rather than like a
cascade. Audio goes up continuously and reply audio comes back concurrently;
**turn detection, barge-in and prosody are owned by the upstream model**, which
hears the interruption directly instead of inferring it from a transcript.

Consequences that are deliberate, not oversights
------------------------------------------------
* The shared Silero segmenter is **not run** on this path. Running both would
  produce two competing turn decisions, and the model's own detector — which
  hears prosody, not just energy — is the better one. This is exactly why the
  engine boundary is drawn at the session level rather than at "STT
  replacement": turn-taking authority moves with the engine.
* `barge_in.classify_utterance()` is likewise unused here. The model knows it
  was interrupted because it heard the interruption.
* The Windows text-overlap echo guard **stops working** under this engine: there
  is no locally-known reply text to compare an incoming transcript against in
  time. Windows therefore needs a real canceller (`voice.native_aec` + a
  registered processor) or headphones. This is recorded in the design's Noise
  section and is a real trade of the engine change.

Depth comes from tools, not from the speech model
-------------------------------------------------
The realtime model answers conversationally and handles ordinary technical
explanation itself. Anything that deserves the full routed stack is delegated
through `tools.py` (`ask_reasoner`, `run_code`, `search_workspace`, …). That is
what makes "it can answer anything by voice" true without pretending a fast
conversational model is also the best reasoner.

Failure is always survivable: any upstream error surfaces as
`EngineError(recoverable=True)` and the runner hands over to `StagedEngine` with
the session ledger intact (Property 7).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from typing import AsyncIterator

from app.voice import budget, tools
from app.voice.engine import (
    AudioDelta, EngineError, EnginePreflight, InterruptReason, Phase,
    PhaseChange, SpeechInterrupted, ToolRequest, ToolResult, TranscriptDelta,
    TurnComplete, UsageDelta, VoiceContext, VoiceEvent,
)

log = logging.getLogger("zapthetrick.voice.realtime")

# How long to wait for the upstream session to open before falling back. Short:
# a slow open is indistinguishable from an outage from the user's seat.
OPEN_TIMEOUT_S = 6.0
# Mic audio is int16 PCM at 24 kHz upstream (the realtime API's native rate).
UPSTREAM_RATE = 24_000

# Session instructions. The "verbalise before a slow tool" line is what turns
# tool latency into audible work rather than dead air.
_INSTRUCTIONS = (
    "You are a helpful spoken assistant. Speak naturally and concisely — this "
    "is a voice conversation, not a document. Never read markdown, code fences "
    "or bullet characters aloud.\n"
    "You have tools that reach the user's own workspace, this conversation's "
    "history, a code sandbox, the web, and a stronger reasoning model. Use "
    "`ask_reasoner` for hard technical questions where being right matters more "
    "than being fast. Before any tool that will take more than a moment, say a "
    "short natural phrase so the user knows you are working — for example "
    "'let me check that'. Never announce tool names.\n"
    "If a tool fails, answer as best you can without it and say briefly that "
    "you could not look it up."
)


def _resample_16k_to_24k(pcm16: bytes) -> bytes:
    """Linear-interpolate 16 kHz int16 mono up to 24 kHz.

    The client captures at the segmenter's 16 kHz; upstream wants 24 kHz. A 3:2
    linear resample is cheap and perfectly adequate for speech — this is not a
    mastering path, and doing it here keeps the client's capture identical
    across both engines (so switching engines never touches the mic).
    """
    try:
        import numpy as np
        src = np.frombuffer(pcm16, dtype="<i2")
        if src.size == 0:
            return b""
        n_out = int(src.size * 3 / 2)
        idx = np.linspace(0, src.size - 1, n_out, dtype="float32")
        out = np.interp(idx, np.arange(src.size, dtype="float32"),
                        src.astype("float32"))
        return out.astype("<i2").tobytes()
    except Exception:  # noqa: BLE001 — never drop a turn over a resample
        return pcm16


class RealtimeSession:
    """One upstream speech-native session, presented as a `VoiceSession`."""

    def __init__(self, ctx: VoiceContext, ws, *, model: str) -> None:
        self._ctx = ctx
        self._ws = ws
        self._model = model
        self._events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self._closed = False
        self._seq = 0
        self._chunks = 0
        self._user_text = ""
        self._assistant_text: list[str] = []
        self._tools_used: list[str] = []
        self._interrupted = False
        # Truncation state. The model generates its transcript AHEAD of
        # playback, so at the moment of a barge-in we hold text the user never
        # heard. Both of these are needed to cut it honestly.
        self._vad_downgraded = False
        self._item_id = ""          # upstream id of the assistant message
        self._audio_ms = 0.0        # audio RECEIVED for the current reply
        self._meter = budget.SessionMeter()
        self._started = time.monotonic()
        self._pump = asyncio.create_task(self._read_upstream())
        self._tool_tasks: set[asyncio.Task] = set()


    def _turn_detection(self) -> dict:
        """Turn detection, as ChatGPT-like as the endpoint allows.

        `semantic_vad` lets the MODEL judge whether a thought is finished rather
        than counting silence. That distinction is why a natural mid-sentence
        pause — "so the thing is… uh…" — does not get treated as the end of your
        turn. A fixed silence timer interrupts anyone who thinks while speaking,
        which is the single most noticeable way a voice assistant feels worse
        than ChatGPT.
        """
        from app.core.config_loader import cfg
        v = cfg.voice
        kind = str(getattr(v, "realtime_turn_detection", "semantic_vad")
                   or "semantic_vad").strip().lower()
        if self._vad_downgraded or kind != "semantic_vad":
            return {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": int(
                    getattr(v, "realtime_silence_ms", 500) or 500),
                "create_response": True,
                # Upstream owns barge-in: it truncates its own reply the moment
                # it hears the user, which is the whole point of moving turn
                # detection to the engine.
                "interrupt_response": True,
            }
        return {
            "type": "semantic_vad",
            "eagerness": str(getattr(v, "realtime_eagerness", "auto")
                             or "auto").strip().lower(),
            "create_response": True,
            "interrupt_response": True,
        }

    def _session_payload(self, ctx: VoiceContext) -> dict:
        from app.core.config_loader import cfg
        v = cfg.voice
        session: dict = {
            "modalities": ["text", "audio"],
            "instructions": _INSTRUCTIONS,
            "voice": (ctx.voice_id or "alloy"),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": str(getattr(v, "realtime_transcribe_model", "whisper-1")
                             or "whisper-1"),
            },
            "turn_detection": self._turn_detection(),
            "tools": tools.schema_for(ctx.allowed_tools or None),
            "tool_choice": "auto",
        }
        nr = str(getattr(v, "realtime_noise_reduction", "") or "").strip().lower()
        if nr in ("near_field", "far_field"):
            # Upstream mic noise suppression. Without it a laptop fan or a room
            # is fed to the turn detector as if it were speech.
            session["input_audio_noise_reduction"] = {"type": nr}
        return session

    async def configure(self) -> None:
        """(Re)send the session configuration."""
        await self._send({"type": "session.update",
                          "session": self._session_payload(self._ctx)})

    async def _maybe_downgrade_vad(self, detail: str) -> bool:
        """Fall back to `server_vad` when the endpoint rejects semantic VAD.

        This engine cannot be validated against a live endpoint from here, and
        an endpoint that does not know `semantic_vad` would otherwise fail the
        whole session over a turn-detection preference. Downgrading keeps the
        conversation alive on the slightly worse setting instead — and says so
        in the log rather than silently pretending the good one is in use.
        """
        if self._vad_downgraded:
            return False
        low = detail.lower()
        if "semantic_vad" not in low and "turn_detection" not in low:
            return False
        self._vad_downgraded = True
        log.warning("realtime: endpoint rejected semantic_vad (%s) — "
                    "falling back to server_vad for this session", detail[:160])
        await self.configure()
        return True

    # ── contract ────────────────────────────────────────────────────────────

    async def send_audio(self, pcm16: bytes) -> None:
        if self._closed or not pcm16:
            return
        payload = base64.b64encode(_resample_16k_to_24k(pcm16)).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": payload})

    async def send_text(self, text: str) -> None:
        t = (text or "").strip()
        if self._closed or not t:
            return
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": t}]},
        })
        await self._send({"type": "response.create"})

    def events(self) -> AsyncIterator[VoiceEvent]:
        async def _iter():
            while True:
                ev = await self._events.get()
                if ev is None:
                    return
                yield ev
        return _iter()

    async def interrupt(self, reason: InterruptReason,
                        played_ms: float | None = None,
                        played_chunks: int | None = None) -> None:
        """Client-initiated barge-in (orb tap). Speech-initiated barge-in comes
        from upstream instead and arrives through `events()`.

        `played_ms` is how much reply audio the client actually rendered. Only
        the client knows this — the server has forwarded more than was heard
        whenever anything sat in the jitter buffer — so when it is reported we
        truncate to it, and fall back to received-audio otherwise.
        """
        if self._closed:
            return
        self._interrupted = True
        await self._send({"type": "response.cancel"})
        await self._truncate_to_heard(played_ms)
        await self._emit(SpeechInterrupted(int(time.monotonic() * 1000), reason))

    async def _truncate_to_heard(self, played_ms: float | None = None) -> None:
        """Cut the reply at the point the user stopped hearing it.

        Two things have to happen, and skipping either produces a specific bug:

        1. **Tell upstream.** Without `conversation.item.truncate` the model's
           context still contains the whole reply, so it believes it said things
           the user never heard — and every follow-up reasons from that false
           premise ("as I mentioned…" about something it did not say). This is
           the half that matters most for conversation quality.
        2. **Cut our own transcript.** The audio deltas we already emitted are
           ahead of playback, so the accumulated transcript contains unspoken
           text. Showing it violates the rule that the user never reads text
           they did not hear.

        Truncation is proportional: transcript text and audio advance together
        within a reply, so the heard FRACTION of the audio is the heard fraction
        of the text. It is an approximation — the alternative is per-word
        timestamps the API does not give us — but it is bounded by a sentence
        and always errs toward showing less rather than inventing more.
        """
        heard = self._audio_ms if played_ms is None else max(0.0, played_ms)
        total = self._audio_ms
        if self._item_id and heard > 0:
            await self._send({
                "type": "conversation.item.truncate",
                "item_id": self._item_id,
                "content_index": 0,
                "audio_end_ms": int(heard),
            })
        if total > 0 and heard < total:
            text = "".join(self._assistant_text)
            keep = int(len(text) * (heard / total))
            self._assistant_text = [text[:keep]]
        self._audio_ms = min(self._audio_ms, heard)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in list(self._tool_tasks):
            t.cancel()
        self._tool_tasks.clear()
        self._pump.cancel()
        with contextlib.suppress(Exception):
            await self._ws.close()
        with contextlib.suppress(Exception):
            self._events.put_nowait(None)

    # ── metering ────────────────────────────────────────────────────────────

    @property
    def meter(self) -> budget.SessionMeter:
        return self._meter

    def over_session_cap(self) -> bool:
        cap = budget.session_seconds_cap()
        return cap > 0 and (time.monotonic() - self._started) >= cap

    # ── internals ───────────────────────────────────────────────────────────

    async def _send(self, obj: dict) -> None:
        try:
            await self._ws.send(json.dumps(obj))
        except Exception as exc:  # noqa: BLE001
            log.info("realtime send failed: %s", exc)
            await self._fail("transport", str(exc))

    async def _emit(self, ev: VoiceEvent) -> None:
        if not self._closed:
            await self._events.put(ev)

    async def _fail(self, kind: str, detail: str) -> None:
        """Every upstream failure is recoverable — the runner hands over to the
        staged engine rather than ending the conversation."""
        await self._emit(EngineError(kind, detail[:400], recoverable=True))

    async def _read_upstream(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    ev = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._fail("upstream_closed", str(exc))

    async def _handle(self, ev: dict) -> None:  # noqa: C901 — a flat dispatch
        kind = str(ev.get("type") or "")

        # Reply audio.
        if kind in ("response.audio.delta", "response.output_audio.delta"):
            b64 = ev.get("delta") or ""
            if b64:
                try:
                    pcm = base64.b64decode(b64)
                except Exception:  # noqa: BLE001
                    return
                # Upstream's id for the assistant message, needed to tell it
                # where the user actually stopped hearing us.
                self._item_id = str(ev.get("item_id") or self._item_id)
                # 24 kHz mono int16 upstream ⇒ 48 bytes per millisecond.
                self._audio_ms += len(pcm) / 48.0
                seq = self._seq
                self._seq += 1
                self._chunks += 1
                await self._emit(PhaseChange(Phase.SPEAKING))
                # Upstream returns raw PCM, not MP3 — the sink appends it
                # directly, which is exactly the seamless path we want.
                await self._emit(AudioDelta(seq, pcm, mp3=False))
            return

        # Assistant transcript (what it is saying).
        if kind in ("response.audio_transcript.delta",
                    "response.output_audio_transcript.delta"):
            d = str(ev.get("delta") or "")
            if d:
                self._assistant_text.append(d)
                await self._emit(TranscriptDelta("assistant", d, final=False))
            return

        # User transcript (what upstream heard).
        if kind == "conversation.item.input_audio_transcription.completed":
            t = str(ev.get("transcript") or "").strip()
            if t:
                self._user_text = t
                await self._emit(TranscriptDelta("user", t, final=True))
            return

        # NATIVE barge-in: upstream heard the user start talking over the reply.
        if kind == "input_audio_buffer.speech_started":
            self._interrupted = True
            await self._truncate_to_heard()
            await self._emit(SpeechInterrupted(int(time.monotonic() * 1000),
                                               InterruptReason.ENGINE))
            return

        if kind == "input_audio_buffer.speech_stopped":
            await self._emit(PhaseChange(Phase.THINKING))
            return

        # Tool call.
        if kind == "response.function_call_arguments.done":
            task = asyncio.create_task(self._run_tool(
                str(ev.get("call_id") or ""), str(ev.get("name") or ""),
                ev.get("arguments") or "{}"))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)
            return

        # Turn finished.
        if kind == "response.done":
            await self._meter_response(ev)
            await self._finish_turn()
            return

        if kind == "error":
            err = ev.get("error") or {}
            detail = str(err.get("message") or "upstream error")
            # A rejected turn-detection setting is a preference problem, not an
            # outage — retry on the fallback rather than ending the session.
            if await self._maybe_downgrade_vad(detail):
                return
            await self._fail(str(err.get("type") or "upstream"), detail)
            return

    async def _run_tool(self, call_id: str, name: str, arguments) -> None:
        """Dispatch one tool and hand the result back to the model.

        The phase goes to TOOL so the client shows work in progress rather than
        silence, and a failure is reported to the MODEL (not the user) so it can
        answer without that tool.
        """
        if not call_id or not name:
            return
        await self._emit(PhaseChange(Phase.TOOL))
        await self._emit(ToolRequest(call_id, name, {}))
        result = await tools.dispatch(name, arguments, self._ctx,
                                      self._ctx.allowed_tools or None)
        ok = "error" not in result
        if name not in self._tools_used:
            self._tools_used.append(name)
        await self._emit(ToolResult(call_id, name, ok,
                                    "" if ok else str(result.get("error"))))
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id,
                     "output": json.dumps(result)[:16000]},
        })
        # Let the model continue speaking with the tool result in hand.
        await self._send({"type": "response.create"})

    async def _meter_response(self, ev: dict) -> None:
        usage = ((ev.get("response") or {}).get("usage")) or {}
        if not usage:
            return
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        cached = int(((usage.get("input_token_details") or {})
                      .get("cached_tokens")) or 0)
        # Cached input is billed far cheaper, so it must not be counted twice.
        in_tok = max(0, in_tok - cached)
        self._meter.add(input_tokens=in_tok, output_tokens=out_tok,
                        cached_tokens=cached)
        await self._emit(UsageDelta(input_tokens=in_tok, output_tokens=out_tok))

    async def _finish_turn(self) -> None:
        assistant = "".join(self._assistant_text).strip()
        user = self._user_text
        if user or assistant:
            await self._emit(TurnComplete(user=user, assistant=assistant,
                                          interrupted=self._interrupted,
                                          chunks=self._chunks))
        await self._emit(PhaseChange(Phase.LISTENING))
        self._user_text = ""
        self._assistant_text = []
        self._interrupted = False
        self._chunks = 0
        self._item_id = ""
        self._audio_ms = 0.0


class RealtimeEngine:
    """Speech-native cloud engine. Network-only — loads no local model."""

    name = "realtime"

    def preflight(self) -> EnginePreflight:
        """Credentials + model + budget. Performs NO network I/O and loads
        nothing, so it cannot delay session start (Requirement 6.4)."""
        from app.voice import policy
        if not policy.realtime_model():
            return EnginePreflight(False, "no realtime model configured")
        if not policy.realtime_credential():
            return EnginePreflight(False, "no realtime credential")
        ok, why = budget.can_open_session()
        if not ok:
            return EnginePreflight(False, why)
        return EnginePreflight(True, turn_detection="upstream_vad")

    async def open(self, ctx: VoiceContext) -> RealtimeSession:
        from app.core.config_loader import cfg
        from app.voice import policy

        model = policy.realtime_model()
        key = policy.realtime_credential()
        base = str(getattr(cfg.voice, "realtime_base_url", "") or
                   "wss://api.openai.com/v1/realtime").rstrip("/")
        url = f"{base}?model={model}"

        import websockets
        ws = await asyncio.wait_for(
            websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Bearer {key}",
                    "OpenAI-Beta": "realtime=v1",
                },
                max_size=None,          # audio deltas are large
                ping_interval=20,
            ),
            timeout=OPEN_TIMEOUT_S,
        )

        session = RealtimeSession(ctx, ws, model=model)
        # Configure the session up front: voice, formats, upstream turn
        # detection, and the tool schema. Instructions + tools are identical
        # every session, which is what makes prompt caching effective.
        await session.configure()
        # Seed prior turns so a handover-in starts with the history rather than
        # an empty head (Property 7).
        for user, assistant in (ctx.history or ())[-6:]:
            for role, text in (("user", user), ("assistant", assistant)):
                if not text:
                    continue
                await session._send({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": role,
                             "content": [{"type": ("input_text"
                                                   if role == "user"
                                                   else "text"),
                                          "text": text[:4000]}]},
                })
        return session


__all__ = ["RealtimeEngine", "RealtimeSession", "UPSTREAM_RATE"]
