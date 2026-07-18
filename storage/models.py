"""SQLAlchemy ORM matching DataBaseArchitecture.md §"Concrete schema".

Schema lives in Postgres. Vectors live in Qdrant (collections keyed by
`{collection}_{owner_id}`); these tables hold the pointers via
`vector_point_id`. Blobs live on the filesystem; tables hold the path
via `file_path`.

Conventions:
  - `id` is always a UUID, server-generated via `uuid_generate_v4`
    (Postgres's `uuid-ossp` extension — see migrations/init/00-extensions.sql).
  - Timestamps are `TIMESTAMPTZ` (`DateTime(timezone=True)` on the
    ORM side) so we never lose UTC information.
  - JSON columns are `JSONB` (faster filtering, indexable).
  - Free-text columns we want to BM25/FTS-search get a
    `content_tsv` generated column with a GIN index.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    synonym,
)


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


def _now_col(**kw) -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="now()",
        **kw,
    )


# ---- Users ---------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = _now_col()
    preferences: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="'{}'::jsonb"
    )


# ---- Resumes ------------------------------------------------------------
class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False, default="Resume")
    profile: Mapped[dict] = mapped_column(JSONB, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    uploaded_at: Mapped[datetime] = _now_col()
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)

    chunks: Mapped[list["ResumeChunk"]] = relationship(
        back_populates="resume",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_resumes_user_active", "user_id", "active"),
    )


class ResumeChunk(Base):
    __tablename__ = "resume_chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    section_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    vector_point_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Generated `tsvector` for BM25 / FTS. The DDL is `to_tsvector('english', content)`.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    resume: Mapped["Resume"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_resume_chunks_resume", "resume_id"),
        Index("ix_resume_chunks_parent", "parent_id"),
        Index("ix_resume_chunks_tsv", "content_tsv", postgresql_using="gin"),
    )


# ---- Sessions + messages ------------------------------------------------
class Project(Base):
    """A project groups conversations (sessions) and scopes their graphs
    (Architecture §17).

    Conversations assigned to a project share:
      - **project-level instructions** injected into every project chat (below
        the safety boundary, alongside the user's own custom instructions), and
      - a **project-scoped knowledge graph** kept in `project_metadata['kg']`, so
        entities/relations accrete across every conversation in the project
        instead of being siloed per session.
    """
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, default="New project")
    # Standing, project-wide instructions (tone/scope/conventions).
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Project-scoped store: `['kg']` holds the merged knowledge graph, and any
    # future project KB pointers. Column named "metadata" to mirror Session.
    project_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="'{}'::jsonb"
    )
    archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default="now()", onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_projects_user_updated", "user_id", "updated_at"),
    )


class Session(Base):
    """One chat / live / solve session.

    Replaces the legacy `conversations` table. The `type` discriminator
    plus the `metadata` JSONB carry everything the old conversation
    record did and more.
    """
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    # §17: the project this conversation belongs to (None = ungrouped). Scopes
    # the knowledge graph + project-level instructions for the turn.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    type: Mapped[str] = mapped_column(String(20), nullable=False, default="chat")
    title: Mapped[str] = mapped_column(Text, nullable=False, default="New session")
    started_at: Mapped[datetime] = _now_col()
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="now()",
        # Server-side (DB) timestamp — NOT Python's naive `datetime.utcnow`,
        # which, written to a `timestamptz` column while the Postgres session
        # runs in a non-UTC timezone, gets shifted by the local offset and
        # stored in the past (the "brand-new chat shows 5h ago" bug at UTC+5:30).
        onupdate=func.now(),
    )
    session_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="'{}'::jsonb",
    )
    # Chat-history affordances (Architecture2.md §"Chat tab — history").
    # Pin/archive let the user curate the drawer; tags drive filtering;
    # message_count/last_message_at let the list render without a join.
    pinned: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Rolling summary of the OLDER part of the conversation, so a very long
    # thread stays within the model's context window (see app/chat/history.py).
    # `summary_count` = how many of the oldest messages the summary covers.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    __table_args__ = (
        Index("ix_sessions_user_pinned_updated", "user_id", "pinned", "updated_at"),
        Index("ix_sessions_user_archived_updated", "user_id", "archived", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    agents_used: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # The unified `response.v1` envelope for this turn (Architecture.md §5),
    # persisted so a reload reconstructs the SAME canonical object it streamed
    # live. Additive + nullable — older rows are None and the load path
    # reconstructs a minimal envelope from the other columns.
    envelope: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # True when this assistant turn was cut short — the client disconnected /
    # the user hit Stop, or the provider dropped mid-stream. The UI surfaces a
    # "Continue / Retry" affordance for these so the partial answer is
    # obviously resumable.
    incomplete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    created_at: Mapped[datetime] = _now_col()
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    session: Mapped["Session"] = relationship(back_populates="messages")

    # Legacy alias — the v1 schema called this `conversation_id`. Keep it
    # as a synonym so old route code (and the Flutter client's `conversation_id`
    # field name) keeps working without touching every call site.
    conversation_id = synonym("session_id")
    # Same trick for `Session` access: `m.conversation` continues to work.
    conversation = synonym("session")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("ix_messages_tsv", "content_tsv", postgresql_using="gin"),
    )


class AgentStep(Base):
    """One event in a Code-In agent run — the ordered, replayable trace.

    Mirrors how `Message` rows hang off a `Session`, but at finer grain: every
    SSE event the agent loop emits (thought / tool_call / tool_result / approval
    / final / error / goal_* / skill) becomes one row, so a past run replays
    EXACTLY. Kept OUT of `messages` so listing sessions stays a no-join query and
    these high-cardinality rows never bloat the message FTS index. The owning
    `Session` has `type="agent_code"`.
    """
    __tablename__ = "agent_steps"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The user-turn Message this step belongs to (null for pre-turn / system).
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    turn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event: Mapped[str] = mapped_column(String(24), nullable=False)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool: Mapped[str | None] = mapped_column(Text, nullable=True)
    # native | mcp | subagent | skill — drives the distinct FE cards.
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="'{}'::jsonb"
    )
    # Wall-clock since the previous step — drives the "Thought for Xs" indicator.
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    incomplete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    created_at: Mapped[datetime] = _now_col()

    __table_args__ = (
        Index("ix_agent_steps_session_seq", "session_id", "seq"),
    )


# ---- Feedback -----------------------------------------------------------
class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=True,
    )
    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    signal: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _now_col()


# ---- Agent runs ---------------------------------------------------------
class AgentRun(Base):
    """One scheduler-tick run of a single agent. Drives the trace view
    and any cost / latency analytics."""
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    agent: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = _now_col()
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    input_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_estimate: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)


# ---- Model usage --------------------------------------------------------
class ModelUsage(Base):
    """Per-call billing ledger. Per spec: provider / model / role +
    prompt+completion tokens + latency."""
    __tablename__ = "model_usage"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occurred_at: Mapped[datetime] = _now_col()


# ---- Episodic + semantic memory ----------------------------------------
class Episode(Base):
    """One completed Q&A turn — episodic memory.

    Vectors live in Qdrant (`episodic_memory_{user_id}` collection); the
    pointer is `vector_point_id`. Token-overlap search still works
    when Qdrant is offline, falling back to `question_embedding_json`
    if it was inlined.
    """
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    # §17: the project this turn belonged to (None = ungrouped). Lets memory
    # recall scope across every conversation in a project (not just one session).
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    session_tag: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    draft: Mapped[str] = mapped_column(Text, nullable=False, default="")
    final: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(50), nullable=False, default="general")
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tools_called: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feedback: Mapped[str | None] = mapped_column(String(10), nullable=True)
    feedback_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    vector_point_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    # Inlined embedding (Qdrant fallback). Empty when Qdrant is healthy
    # — saves a few KB per row.
    question_embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _now_col()


class SolveSession(Base):
    """One Solve-screen click → one row.

    Captures everything needed to reload a past solve from the history
    drawer: the user-facing title (so the list reads well), the full
    problem statement (typed text or OCR'd from the screenshot), and
    the model's response. The image blob path is set when the source
    was a screenshot — the bytes themselves live in the BlobStore.

    Indexed by `created_at DESC` so the most-recent solve always tops
    the history list.
    """
    __tablename__ = "solve_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="Untitled solve")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False, default="")
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="text")  # text | image
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    vision_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now_col()

    __table_args__ = (
        Index("ix_solve_sessions_created_at", "created_at"),
    )


class SkillRow(Base):
    """Distilled lesson from the Reflector — semantic memory.

    Vectors live in Qdrant (`semantic_memory_{user_id}`); the pointer
    is `vector_point_id`. The inlined embedding stays as a fallback.
    """
    __tablename__ = "skills"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    # §17: project scope for cross-conversation skill recall (None = ungrouped).
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    session_tag: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="preference")
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)
    evidence_episode_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    vector_point_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    text_embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _now_col()


# ── Multi-provider LLM routing (freellmapi port) ─────────────────────────
# These six tables back `app/llm/*`. Integer PKs (not UUID) so the router's
# round-robin + fallback references stay cheap and mirror the reference
# implementation. See migration 0005_llm_routing.


class LLMApiKey(Base):
    """One encrypted provider API key. Many keys per platform are allowed —
    the router round-robins across the enabled, healthy ones."""
    __tablename__ = "llm_api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    iv: Mapped[str] = mapped_column(Text, nullable=False)
    auth_tag: Mapped[str] = mapped_column(Text, nullable=False)
    # unknown | healthy | invalid | error
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="unknown")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    # Consecutive confirmed-invalid validations; 3 → auto-disable (health.py).
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = _now_col()
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_llm_api_keys_platform", "platform"),)


class LLMModel(Base):
    """Catalog row: one routable model on one platform, with the free-tier
    rate limits and intelligence/speed ranks that drive routing."""
    __tablename__ = "llm_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    intelligence_rank: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    speed_rank: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    size_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rpd_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tpd_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_token_budget: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    # Multimodal: True when the model accepts image input, so the router can
    # send image-bearing chat turns only to vision-capable models.
    supports_vision: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    __table_args__ = (
        Index("ix_llm_models_platform_model", "platform", "model_id", unique=True),
    )


class LLMFallbackConfig(Base):
    """Priority chain. Lower `priority` is tried first. The router adds a
    dynamic penalty on top so rate-limited models sink automatically."""
    __tablename__ = "llm_fallback_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_db_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("llm_models.id", ondelete="CASCADE"), nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    __table_args__ = (
        Index("ix_llm_fallback_model", "model_db_id", unique=True),
        Index("ix_llm_fallback_priority", "priority"),
    )


class LLMRateLimitUsage(Base):
    """Append-only usage ledger for the sliding-window rate limiter.
    Rows older than 24h are pruned on write (see ratelimit.py)."""
    __tablename__ = "llm_rate_limit_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    key_id: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False)  # request | tokens
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("ix_llm_rl_usage_lookup", "platform", "model_id", "key_id", "kind", "created_at_ms"),
    )


class LLMRateLimitCooldown(Base):
    """Per (platform, model, key) cooldown set after a 429. Survives restarts
    so a daily-quota exhaustion stays quarantined."""
    __tablename__ = "llm_rate_limit_cooldowns"

    platform: Mapped[str] = mapped_column(Text, primary_key=True)
    model_id: Mapped[str] = mapped_column(Text, primary_key=True)
    key_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    expires_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (Index("ix_llm_rl_cooldowns_expires", "expires_at_ms"),)


class LLMSetting(Base):
    """Small key/value store for the routing subsystem (dev encryption key,
    unified knobs). Distinct from config.yaml — these are runtime secrets."""
    __tablename__ = "llm_settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class GeneratedDocument(Base):
    """A versioned generated-document artifact (Document Generation roadmap,
    Phase 5 lifecycle). Stores the SOURCE Markdown — the structured DocumentModel
    and every export format are derived from it on demand, so one row backs all
    renderings. An edit ("add a Redis section", "convert to Word") creates a NEW
    row: same ``doc_key``, ``version`` + 1 — giving a full evolution timeline
    without duplicating unrelated turns."""
    __tablename__ = "generated_documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Legacy alias so route code using `conversation_id` keeps working.
    conversation_id = synonym("session_id")
    # Groups every version of the SAME logical document.
    doc_key: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    doc_format: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pdf")
    goal: Mapped[str | None] = mapped_column(String(40), nullable=True)
    content_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _now_col()

    __table_args__ = (
        Index("ix_generated_documents_key_version", "doc_key", "version"),
    )
