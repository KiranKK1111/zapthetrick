"""Persistent per-user Q&A log.

Backed by Postgres (`episodes` table — source of truth) + the
configured VectorStore (`episodic_memory_{user_id}` collection — fast
similarity index). The inline JSONB embedding is kept as a fallback so
search still works if the vector store is offline.

Why both layers:
  - Postgres = audit trail, filters, feedback joins, FTS — durable.
  - Qdrant   = sub-millisecond similarity at any scale — derived index.

DataBaseArchitecture.md §"Transactions and consistency":
  Postgres writes first, then the vector upsert. On vector failure the
  row stays committed; a background reindex can fill it later.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import Episode as EpisodeRow


log = logging.getLogger(__name__)

# Default user suffix when no user_id is available (single-user-on-device).
_DEFAULT_USER_TAG = "default"


def _collection_for(user_id: uuid.UUID | str | None) -> str:
    return f"episodic_memory_{user_id or _DEFAULT_USER_TAG}"


def _as_uuid(v):
    if v is None or isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (TypeError, ValueError):
        return None


@dataclass
class Episode:
    """One completed Q&A turn with feedback + provenance."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""                 # legacy alias for session_tag
    user_id: str | None = None           # device user (account-level scoping §18)
    project_id: str | None = None        # project scope (§17); None = ungrouped
    question: str = ""
    draft: str = ""
    final: str = ""
    intent: str = "general"
    sources: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    latency_ms: int = 0
    feedback: str | None = None
    feedback_payload: dict | None = None
    question_embedding: list[float] | None = None
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class EpisodicMemory:
    """In-process episodic log — tests + transient sessions."""

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def record(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def attach_feedback(
        self, episode_id: str, kind: str, payload: dict | None = None
    ) -> None:
        for ep in self._episodes:
            if ep.id == episode_id:
                ep.feedback = kind
                ep.feedback_payload = payload
                return

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def search_similar(self, question: str, *, top_k: int = 3) -> list[Episode]:
        return _rank_by_overlap(self._episodes, question, top_k=top_k)


# ---- Postgres + VectorStore helpers (prod) -------------------------------
async def record_episode(session: AsyncSession, episode: Episode) -> str:
    """Persist to Postgres, then upsert into the Qdrant collection.

    Vector upsert failures are non-fatal — the row stays committed; the
    embedding remains on the Postgres JSONB column as a fallback.
    """
    embedding: list[float] | None = None
    try:
        from ..rag import embedder as _embedder

        embedding = await asyncio.to_thread(_embedder.embed_one, episode.question)
    except Exception:
        embedding = None

    vector_point_id = uuid.uuid4()

    row = EpisodeRow(
        user_id=_as_uuid(episode.user_id),
        project_id=_as_uuid(episode.project_id),
        session_tag=episode.session_id,
        question=episode.question,
        draft=episode.draft,
        final=episode.final,
        intent=episode.intent,
        sources=list(episode.sources) if episode.sources else None,
        tools_called=list(episode.tools_called) if episode.tools_called else None,
        latency_ms=episode.latency_ms,
        feedback=episode.feedback,
        feedback_payload=episode.feedback_payload,
        question_embedding=embedding,
        vector_point_id=vector_point_id if embedding else None,
    )
    session.add(row)
    await session.commit()
    row_id = str(row.id)

    # Vector upsert — fire-and-forget on failure.
    if embedding:
        await _vector_upsert_episode(row, embedding)

    return row_id


async def _vector_upsert_episode(row: EpisodeRow, embedding: list[float]) -> None:
    """Push the embedding into the per-user Qdrant collection.

    Best-effort: a vector store outage doesn't fail the turn — Postgres
    still has the row and the inline JSONB embedding.
    """
    try:
        from app.core.config_loader import cfg
        from storage.vectors import get_vector_store

        store = get_vector_store()
        collection = _collection_for(getattr(row, "user_id", None))
        await store.ensure_collection(
            collection,
            vector_size=len(embedding),
            embedding_model=cfg.embeddings.model,
        )
        await store.upsert(
            collection,
            ids=[str(row.vector_point_id)],
            vectors=[embedding],
            payloads=[
                {
                    "episode_id": str(row.id),
                    "session_tag": row.session_tag,
                    "project_id": str(row.project_id) if row.project_id else None,
                    "intent": row.intent,
                    "question": row.question,
                }
            ],
        )
    except Exception as exc:
        log.warning("episodic vector upsert deferred: %s", exc)


async def attach_feedback_db(
    session: AsyncSession,
    episode_id: str,
    kind: str,
    payload: dict | None = None,
) -> None:
    try:
        ep_uuid = uuid.UUID(episode_id)
    except (TypeError, ValueError):
        return
    row = await session.get(EpisodeRow, ep_uuid)
    if row is None:
        return
    row.feedback = kind
    row.feedback_payload = payload
    await session.commit()


async def recent_episodes(
    session: AsyncSession,
    *,
    session_id: str | None = None,
    limit: int = 50,
) -> list[Episode]:
    stmt = select(EpisodeRow).order_by(EpisodeRow.created_at.desc()).limit(limit)
    if session_id is not None:
        stmt = (
            select(EpisodeRow)
            .where(EpisodeRow.session_tag == session_id)
            .order_by(EpisodeRow.created_at.desc())
            .limit(limit)
        )
    result = await session.execute(stmt)
    return [_row_to_episode(r) for r in result.scalars().all()]


async def search_episodes_similar(
    session: AsyncSession,
    question: str,
    *,
    session_id: str | None = None,
    top_k: int = 3,
    user_id: uuid.UUID | str | None = None,
    project_id: uuid.UUID | str | None = None,
) -> list[Episode]:
    """Cosine-rank past episodes against `question`.

    §17: when `project_id` is given, recall scopes to the whole PROJECT (every
    conversation in it) rather than the single session — so a project chat sees
    what was learned in its sibling chats. Otherwise it scopes by `session_id`
    as today.

    Path priority:
      1. **Vector store** (Qdrant / Chroma) — HNSW ANN at any scale.
      2. **Postgres inline JSONB** — fallback when the vector store is
         offline or empty for this user.
      3. **Token overlap** — final fallback when the embedder itself
         is unavailable.
    """
    if not question.strip():
        return []

    # 1. Try the vector store.
    query_embedding: list[float] | None = None
    try:
        from ..rag import embedder as _embedder

        query_embedding = await asyncio.to_thread(_embedder.embed_one, question)
    except Exception:
        query_embedding = None

    if query_embedding:
        hits = await _vector_search_episodes(
            query_embedding, user_id=user_id, session_id=session_id,
            project_id=project_id, k=top_k
        )
        if hits:
            return await _hits_to_episodes(session, hits)

    # 2. JSONB fallback — pull recent rows and rank in Python.
    return await _jsonb_fallback_search(
        session,
        question,
        query_embedding=query_embedding,
        session_id=session_id,
        project_id=project_id,
        top_k=top_k,
    )


async def _vector_search_episodes(
    query_embedding: list[float],
    *,
    user_id: uuid.UUID | str | None,
    session_id: str | None,
    k: int,
    project_id: uuid.UUID | str | None = None,
):
    """Per-user collection lookup. Empty list on any error / missing collection."""
    try:
        from app.core.config_loader import cfg
        from storage.vectors import get_vector_store

        store = get_vector_store()
        collection = _collection_for(user_id)
        await store.ensure_collection(
            collection,
            vector_size=len(query_embedding),
            embedding_model=cfg.embeddings.model,
        )
        # Project scope wins over session scope (§17).
        if project_id is not None:
            flt = {"project_id": str(project_id)}
        elif session_id is not None:
            flt = {"session_tag": session_id}
        else:
            flt = None
        return await store.query(collection, vector=query_embedding, k=k, filter=flt)
    except Exception as exc:
        log.debug("vector search fell through to JSONB: %s", exc)
        return []


async def _hits_to_episodes(session: AsyncSession, hits) -> list[Episode]:
    """Resolve vector hits to Episode rows via `vector_point_id`."""
    point_ids = [uuid.UUID(h.id) for h in hits if h.id]
    if not point_ids:
        return []
    result = await session.execute(
        select(EpisodeRow).where(EpisodeRow.vector_point_id.in_(point_ids))
    )
    by_pid = {r.vector_point_id: r for r in result.scalars().all()}
    # Preserve vector-store ranking order.
    out: list[Episode] = []
    for h in hits:
        try:
            pid = uuid.UUID(h.id)
        except ValueError:
            continue
        row = by_pid.get(pid)
        if row is not None:
            out.append(_row_to_episode(row))
    return out


async def _jsonb_fallback_search(
    session: AsyncSession,
    question: str,
    *,
    query_embedding: list[float] | None,
    session_id: str | None,
    top_k: int,
    project_id: uuid.UUID | str | None = None,
) -> list[Episode]:
    # Bounded small — vector-store-offline fallback that ranks in Python every
    # turn, so it must not grow with the user's episode history.
    stmt = select(EpisodeRow).order_by(EpisodeRow.created_at.desc()).limit(30)
    if project_id is not None:                      # §17: project scope wins
        stmt = (
            select(EpisodeRow)
            .where(EpisodeRow.project_id == _as_uuid(project_id))
            .order_by(EpisodeRow.created_at.desc())
            .limit(30)
        )
    elif session_id is not None:
        stmt = (
            select(EpisodeRow)
            .where(EpisodeRow.session_tag == session_id)
            .order_by(EpisodeRow.created_at.desc())
            .limit(30)
        )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    if not rows:
        return []
    episodes = [_row_to_episode(r) for r in rows]

    if query_embedding:
        q_tokens = set(question.lower().split())
        scored: list[tuple[float, Episode]] = []
        for ep in episodes:
            if ep.question_embedding:
                score = _cosine(query_embedding, ep.question_embedding)
            else:
                ep_tokens = set(ep.question.lower().split())
                overlap = len(q_tokens & ep_tokens)
                score = overlap / max(len(q_tokens | ep_tokens), 1)
            scored.append((score, ep))
        scored.sort(key=lambda kv: -kv[0])
        return [ep for score, ep in scored if score > 0.1][:top_k]

    return _rank_by_overlap(episodes, question, top_k=top_k)


def _row_to_episode(row: EpisodeRow) -> Episode:
    return Episode(
        id=str(row.id),
        session_id=row.session_tag,
        user_id=str(row.user_id) if row.user_id else None,
        project_id=str(row.project_id) if row.project_id else None,
        question=row.question,
        draft=row.draft or "",
        final=row.final,
        intent=row.intent,
        sources=list(row.sources or []),
        tools_called=list(row.tools_called or []),
        latency_ms=row.latency_ms or 0,
        feedback=row.feedback,
        feedback_payload=row.feedback_payload,
        question_embedding=row.question_embedding,
        ts_ms=int(row.created_at.timestamp() * 1000) if row.created_at else 0,
    )


def _rank_by_overlap(
    episodes: list[Episode], question: str, *, top_k: int
) -> list[Episode]:
    q_tokens = set(question.lower().split())
    now_ms = int(time.time() * 1000)
    raw: list[tuple[Episode, float, float]] = []
    for ep in episodes:
        overlap = len(q_tokens & set(ep.question.lower().split()))
        if overlap <= 0:
            continue
        age_s = max(0.0, (now_ms - getattr(ep, "ts_ms", now_ms)) / 1000.0)
        raw.append((ep, float(overlap), age_s))
    if not raw:
        return []
    # Knowledge Freshness (roadmap P3 #17): recalled memories AGE — among
    # similarly-relevant episodes prefer the fresher one, so a stale answer
    # doesn't outrank a recent one. Overlap relevance is normalised, then blended
    # with a recency signal. Fail-open to pure-overlap ordering.
    try:
        from app.rag.freshness import rerank_by_freshness
        max_rel = max(r for _, r, _ in raw) or 1.0
        items = [(i, r / max_rel, a) for i, (_ep, r, a) in enumerate(raw)]
        ranked = rerank_by_freshness(items, freshness_weight=0.15)
        return [raw[idx][0] for idx, _score in ranked][:top_k]
    except Exception:  # noqa: BLE001
        raw.sort(key=lambda t: -t[1])
        return [ep for ep, _r, _a in raw][:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
