"""The `VoiceEngine` / `VoiceSession` contract (design §1).

This supersedes `app/live/s2s.py`. The critical difference is that the contract
is **session-scoped, not turn-scoped**.

The old seam was::

    turn(pcm16: bytes, *, context: dict) -> AsyncIterator[tuple[str, bytes]]

One complete utterance in, one reply stream out. That shape cannot host a
speech-native model for two reasons:

1. It presupposes somebody else already decided where the utterance ended —
   which is precisely the decision a realtime model makes better than a
   server-side VAD.
2. It has no channel for the model to say "the user started talking, I
   stopped", so native barge-in is inexpressible.

Session scope fixes both: audio flows both directions concurrently and
*turn-taking authority moves with the engine*. `StagedEngine` keeps using the
shared Silero segmenter; `RealtimeEngine` bypasses it entirely, because running
two turn detectors would produce two competing decisions.

Everything in this module is pure typing + small dataclasses — importing it
loads no model and opens no socket (design Property 9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable


class Phase(str, Enum):
    """What the orb shows. `TOOL` exists so a slow tool call is audible as
    work-in-progress rather than dead air (design §4)."""

    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TOOL = "tool"
    ERROR = "error"


class InterruptReason(str, Enum):
    TAP = "tap"            # user tapped the orb
    SPEECH = "speech"      # confirmed barge-in from speech
    ENGINE = "engine"      # the engine detected it natively


# ── Event union ──────────────────────────────────────────────────────────────
# A closed union emitted on ONE ordered stream. Ordering is the engine's
# guarantee, not the client's reassembly job — that is what makes design
# Property 2 (audio plays in emission order) hold across a Kokoro→Edge fallback.

@dataclass(frozen=True)
class AudioDelta:
    """One chunk of reply audio. `seq` is monotonic within a turn."""
    seq: int
    payload: bytes
    mp3: bool = True          # False ⇒ raw int16 PCM at protocol.SAMPLE_RATE


@dataclass(frozen=True)
class TranscriptDelta:
    role: str                 # "user" | "assistant"
    text: str
    final: bool = False
    # Semantic "does this partial read as a finished thought" (doc §6
    # Incremental Thinking). The staged client uses it to start answering
    # SPECULATIVELY while the user is still finishing, so the first-token wait
    # hides inside the endpoint silence. Meaningless on a final.
    complete: bool = False


@dataclass(frozen=True)
class SttStatus:
    """A swallowed STT failure/empty, surfaced so a dead mic is EXPLAINABLE
    instead of silently producing nothing. Distinct from `EngineError` because
    it is informational — the session continues."""
    state: str                # "error" | "empty"
    detail: str = ""


@dataclass(frozen=True)
class PhaseChange:
    phase: Phase


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class SpeechInterrupted:
    at_ms: int
    reason: InterruptReason = InterruptReason.ENGINE


@dataclass(frozen=True)
class TurnComplete:
    """Exactly one per turn (design Property 3). `chunks` lets the client assert
    it rendered everything the engine produced."""
    user: str
    assistant: str
    interrupted: bool = False
    chunks: int = 0


@dataclass(frozen=True)
class AudioDropped:
    """Property 1: audio the engine abandoned is reported, never silently lost."""
    seq: int
    reason: str


@dataclass(frozen=True)
class EngineError:
    kind: str
    detail: str
    recoverable: bool = True


@dataclass(frozen=True)
class UsageDelta:
    """Metering as the session proceeds (Requirement 9.3) — not at session end,
    so the budget governor can stop a runaway before it becomes a bill."""
    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0


VoiceEvent = (
    AudioDelta | TranscriptDelta | PhaseChange | ToolRequest | ToolResult
    | SpeechInterrupted | TurnComplete | AudioDropped | EngineError | UsageDelta
    | SttStatus
)


# ── Session context + ledger ─────────────────────────────────────────────────

@dataclass
class VoiceContext:
    """Everything an engine needs to open a session. Built once by the runner."""

    user_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    voice_id: str = ""
    language: str = "en"
    speed: float = 1.0
    allowed_tools: frozenset[str] = frozenset()
    speaker_profile: object | None = None
    # Prior turns, so an engine opened mid-conversation (a handover) starts with
    # the history rather than an empty head. This is what makes Property 7 work.
    history: tuple[tuple[str, str], ...] = ()


@dataclass
class VoiceTurn:
    """One completed exchange, in the shape chat already consumes."""

    user: str
    assistant: str
    engine: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    tools_used: tuple[str, ...] = ()
    interrupted: bool = False


@dataclass
class SessionLedger:
    """Append-only turn history + metering for one socket.

    This is what survives an engine handover: `transcript.py` turns it into chat
    messages on close, so a mid-session failure loses no spoken turn.
    """

    turns: list[VoiceTurn] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0
    engine_switches: list[tuple[str, str, str]] = field(default_factory=list)

    def record(self, turn: VoiceTurn) -> None:
        self.turns.append(turn)

    def meter(self, delta: UsageDelta) -> None:
        self.input_tokens += int(delta.input_tokens or 0)
        self.output_tokens += int(delta.output_tokens or 0)
        self.audio_seconds += float(delta.audio_seconds or 0.0)

    def history(self) -> tuple[tuple[str, str], ...]:
        """Completed turns as (user, assistant) pairs — the handover payload."""
        return tuple((t.user, t.assistant) for t in self.turns
                     if t.user and t.assistant)


@dataclass(frozen=True)
class EnginePreflight:
    """Cheap readiness. MUST NOT load a model or block session start
    (Requirement 6.4)."""

    ok: bool
    reason: str = ""
    # Advisory: how the engine detects turns, surfaced in `session.ready` so the
    # client can show the right affordance.
    turn_detection: str = "server_vad"


# ── The contract ─────────────────────────────────────────────────────────────

@runtime_checkable
class VoiceSession(Protocol):
    """A live, full-duplex conversation. Audio flows both directions
    concurrently; the SESSION decides turn boundaries."""

    async def send_audio(self, pcm16: bytes) -> None:
        """Continuous mic frames. Must never block on the model."""

    async def send_text(self, text: str) -> None:
        """Typed input during a voice session (accessibility path)."""

    def events(self) -> AsyncIterator[VoiceEvent]:
        """Single ordered output stream."""

    async def interrupt(self, reason: InterruptReason) -> None:
        """Client-initiated barge-in. Engines that detect interruption natively
        also raise it through `events()`."""

    async def close(self) -> None:
        ...


@runtime_checkable
class VoiceEngine(Protocol):
    name: str

    async def open(self, ctx: VoiceContext) -> VoiceSession:
        ...

    def preflight(self) -> EnginePreflight:
        """Credentials present, endpoint reachable, budget remaining. Never
        loads a model, never blocks."""


# ── Registry ─────────────────────────────────────────────────────────────────
# Small and explicit. The runner asks `policy.py` which name to use, then
# resolves it here, so engine construction stays out of the selection logic.

_ENGINES: dict[str, VoiceEngine] = {}


def register(engine: VoiceEngine) -> None:
    _ENGINES[engine.name] = engine


def get(name: str) -> VoiceEngine | None:
    return _ENGINES.get(name)


def available() -> tuple[str, ...]:
    return tuple(sorted(_ENGINES))


def clear() -> None:
    """Test hook — the registry is process-global."""
    _ENGINES.clear()


__all__ = [
    "Phase", "InterruptReason", "AudioDelta", "TranscriptDelta", "PhaseChange",
    "ToolRequest", "ToolResult", "SpeechInterrupted", "TurnComplete",
    "SttStatus",
    "AudioDropped", "EngineError", "UsageDelta", "VoiceEvent", "VoiceContext",
    "VoiceTurn", "SessionLedger", "EnginePreflight", "VoiceSession",
    "VoiceEngine", "register", "get", "available", "clear",
]
