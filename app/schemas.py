"""
Pydantic schemas for the API surface.

These define the request and response shapes for the REST endpoints. The
streaming endpoint sends Server-Sent Events whose data payloads are described
in routes.py rather than here, because SSE frames are not single JSON bodies.
"""
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _stringify_uuid(v: Any) -> Any:
    """Pydantic-v2 helper: coerce `uuid.UUID` (and anything stringy) to str.

    The new schema uses `UUID(as_uuid=True)` PKs, so SQLAlchemy returns
    `uuid.UUID` objects. Response models declare `id: str` for the
    Flutter client's convenience — without this coercion Pydantic v2
    raises `Input should be a valid string` and FastAPI returns 500.
    """
    if isinstance(v, uuid.UUID):
        return str(v)
    return v


class ChatRequest(BaseModel):
    """Body for POST /api/chat/stream."""
    message: str = Field(..., min_length=1, description="The user's message.")
    conversation_id: str | None = Field(
        None,
        description="Existing conversation to continue. If omitted, a new "
        "conversation is created and its id is returned in the stream.",
    )
    # Architecture.md §"Depth control". Echoed back through
    # response_arch.finalize so TL;DR / Standard / Deeper / Exhaustive
    # truncate the assistant reply to the matching cap.
    depth: str | None = Field(
        None,
        description="tldr | standard | deeper | exhaustive. None = default.",
    )


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    intent: str | None = None
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return _stringify_uuid(v)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    updated_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return _stringify_uuid(v)


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return _stringify_uuid(v)


class HealthResponse(BaseModel):
    backend: str = "ok"
    database: str
    # The configured LLM provider's status: "ok" or an error reason.
    llm: str
    # Which provider answered: ollama | groq | gemini | anthropic | ...
    provider: str
    # Which model is currently configured.
    model: str


# ---- Resume Q&A --------------------------------------------------------
class ResumeUploadResponse(BaseModel):
    """Returned after a resume is uploaded and parsed."""
    resume_id: str
    display_name: str
    filename: str
    profile: dict
    created_at: datetime

    @field_validator("resume_id", mode="before")
    @classmethod
    def _resume_id_to_str(cls, v):
        return _stringify_uuid(v)


class ResumeSummary(BaseModel):
    """Lightweight row for the resume list."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    filename: str
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return _stringify_uuid(v)


class ResumeDetail(BaseModel):
    """Full resume record, including the parsed profile."""
    id: str
    display_name: str
    filename: str
    profile: dict
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v):
        return _stringify_uuid(v)


class AgentsStreamRequest(BaseModel):
    """Body for POST /api/agents/stream.

    The multi-agent endpoint: a single user message is routed through
    the full mesh (Planner → Retriever → Persona → Grounder, with
    Memory/Critic/Suggester in parallel and Reflector after).

    Persistence:
      - If `conversation_id` is set, append to that conversation.
      - Otherwise a new conversation is created and its id flows back
        on the `meta` SSE event (same shape as /api/chat/stream).
      - `resume_id` opts the Retriever in; without it, the Persona
        agent falls back to its system prompt alone.
      - `session_id` keys the episodic memory log.
    """
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    resume_id: str | None = None
    session_id: str | None = Field(
        None,
        description="Persistent session id; reuse to thread follow-up context.",
    )
    # Architecture.md §"Depth control" — feeds response_arch.finalize.
    depth: str | None = Field(
        None,
        description="tldr | standard | deeper | exhaustive. None = default.",
    )
    # Optional manual difficulty/effort override from the UI ("think harder").
    # None → the difficulty classifier decides (normal). When set to a valid
    # level it OVERRIDES the classifier for this turn, routing to a stronger
    # (or lighter) model. Invalid values are ignored (fail-open to classifier).
    difficulty: str | None = Field(
        None,
        description="trivial | standard | hard | expert. None = auto (classifier).",
    )
    # True when this turn is the answer to a clarification panel — the
    # Clarifier must NOT re-trigger on it (otherwise it asks again in a loop).
    skip_clarify: bool = False
    # Optional per-turn override of the active clarification mode
    # (explorer | builder | expert | autopilot | teacher). Normally the mode is
    # stored per-device and resolved server-side; this lets a client force one.
    clarify_mode: str | None = None
    # Client rendering-load hint (0.0..1.0) — when high, the server coalesces
    # streamed tokens into larger chunks to reduce frame pressure (R46).
    client_load: float | None = None
    # Perceived-speed (R1): the token returned by POST /api/prefetch for work
    # warmed while the user was typing. The route consumes it (reuse) on submit
    # so the warmed connection/handles are used instead of repeated; a mismatch
    # is discarded. Optional/additive — absent → normal request.
    prefetch_token: str | None = None


class FeedbackRequest(BaseModel):
    """Body for POST /api/agents/episodes/{id}/feedback."""
    kind: str = Field(..., description="'up' | 'down' | 'edit' | 'redo'")
    payload: dict | None = None


class ResumeAskRequest(BaseModel):
    """Body for POST /api/resume/ask/stream."""
    resume_id: str = Field(..., description="Resume to answer as.")
    question: str = Field(..., min_length=1, description="The interviewer's question.")
    # Optional client-supplied session id — keys the context tracker for
    # follow-up detection. Defaults to a fresh UUID per request if omitted.
    session_id: str | None = Field(
        None,
        description="Persistent session id; reuse to carry follow-up context.",
    )
