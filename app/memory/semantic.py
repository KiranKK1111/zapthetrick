"""Distilled skills, preferences, and lessons learned about the user.

Backed by Postgres `skills` (source of truth) + the VectorStore
`semantic_memory_{user_id}` collection (fast similarity index). Same
write-through pattern as [episodic.py].
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

from storage.models import SkillRow


log = logging.getLogger(__name__)

_DEFAULT_USER_TAG = "default"


def _collection_for(user_id: uuid.UUID | str | None) -> str:
    return f"semantic_memory_{user_id or _DEFAULT_USER_TAG}"


def _as_uuid(v):
    if v is None or isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (TypeError, ValueError):
        return None


@dataclass
class Skill:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""                   # legacy alias for session_tag
    user_id: str | None = None             # device user (account scoping §18)
    project_id: str | None = None          # project scope (§17); None = ungrouped
    text: str = ""
    kind: str = "preference"
    confidence: float = 0.5
    evidence_episode_ids: list[str] = field(default_factory=list)
    text_embedding: list[float] | None = None
    last_seen_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class SemanticMemory:
    """In-memory registry — tests + short-lived sessions."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def upsert(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def remove(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def relevant_skills(self, question: str, *, top_k: int = 3) -> list[Skill]:
        q_tokens = set(question.lower().split())
        scored = [
            (len(q_tokens & set(s.text.lower().split())), s)
            for s in self._skills.values()
        ]
        scored.sort(key=lambda kv: -kv[0])
        return [s for score, s in scored if score > 0][:top_k]


# ---- Postgres + VectorStore helpers (prod) ------------------------------
async def _embed_or_none(text: str) -> list[float] | None:
    try:
        from ..rag import embedder as _embedder

        vec = await asyncio.to_thread(_embedder.embed_one, text)
        return vec or None
    except Exception:
        return None


async def record_skill(session: AsyncSession, skill: Skill) -> str:
    """De-duped insert with vector upsert. DataBaseArchitecture.md
    §"Transactions and consistency": Postgres first, then vector."""
    existing = await session.execute(
        select(SkillRow).where(
            SkillRow.session_tag == skill.session_id,
            SkillRow.text == skill.text,
        )
    )
    row = existing.scalar_one_or_none()
    if row is not None:
        row.confidence = max(float(row.confidence), skill.confidence)
        evidence = set(row.evidence_episode_ids or [])
        evidence.update(skill.evidence_episode_ids or [])
        row.evidence_episode_ids = sorted(evidence)
        if not row.text_embedding:
            row.text_embedding = await _embed_or_none(skill.text)
            if row.text_embedding and not row.vector_point_id:
                row.vector_point_id = uuid.uuid4()
        await session.commit()
        if row.text_embedding and row.vector_point_id:
            await _vector_upsert_skill(row)
        return str(row.id)

    embedding = await _embed_or_none(skill.text)
    vector_point_id = uuid.uuid4() if embedding else None
    row = SkillRow(
        user_id=_as_uuid(skill.user_id),
        project_id=_as_uuid(skill.project_id),
        session_tag=skill.session_id,
        text=skill.text,
        kind=skill.kind,
        confidence=skill.confidence,
        evidence_episode_ids=list(skill.evidence_episode_ids or []),
        text_embedding=embedding,
        vector_point_id=vector_point_id,
    )
    session.add(row)
    await session.commit()

    if embedding and vector_point_id:
        await _vector_upsert_skill(row)
    return str(row.id)


async def _vector_upsert_skill(row: SkillRow) -> None:
    """Best-effort vector upsert. Postgres has the durable copy."""
    try:
        from app.core.config_loader import cfg
        from storage.vectors import get_vector_store

        store = get_vector_store()
        collection = _collection_for(getattr(row, "user_id", None))
        await store.ensure_collection(
            collection,
            vector_size=len(row.text_embedding or []),
            embedding_model=cfg.embeddings.model,
        )
        await store.upsert(
            collection,
            ids=[str(row.vector_point_id)],
            vectors=[list(row.text_embedding or [])],
            payloads=[
                {
                    "skill_id": str(row.id),
                    "session_tag": row.session_tag,
                    "project_id": str(row.project_id) if row.project_id else None,
                    "kind": row.kind,
                    "text": row.text,
                    "confidence": float(row.confidence),
                }
            ],
        )
    except Exception as exc:
        log.warning("semantic vector upsert deferred: %s", exc)


async def list_skills_for_session(
    session: AsyncSession, session_id: str, *, top_k: int = 20
) -> list[Skill]:
    result = await session.execute(
        select(SkillRow)
        .where(SkillRow.session_tag == session_id)
        .order_by(SkillRow.confidence.desc())
        .limit(top_k)
    )
    return [_row_to_skill(r) for r in result.scalars().all()]


async def relevant_skills_for_question(
    session: AsyncSession,
    question: str,
    *,
    session_id: str | None = None,
    top_k: int = 3,
    user_id: uuid.UUID | str | None = None,
    project_id: uuid.UUID | str | None = None,
) -> list[Skill]:
    """Vector-first, JSONB fallback, token-overlap last resort.

    §17: when `project_id` is given, recall scopes to the whole PROJECT rather
    than the single session."""
    if not question.strip():
        return []

    query_vec: list[float] | None = None
    try:
        from ..rag import embedder as _embedder

        query_vec = await asyncio.to_thread(_embedder.embed_one, question)
    except Exception:
        query_vec = None

    # 1. Vector store path.
    if query_vec:
        hits = await _vector_search_skills(
            query_vec, user_id=user_id, session_id=session_id,
            project_id=project_id, k=top_k
        )
        if hits:
            return await _hits_to_skills(session, hits)

    # 2. JSONB fallback (vector store offline). Bounded small — this runs per
    #    turn and ranks in Python, so it must not grow with history.
    stmt = select(SkillRow).order_by(SkillRow.confidence.desc()).limit(20)
    if project_id is not None:                      # §17: project scope wins
        stmt = (
            select(SkillRow)
            .where(SkillRow.project_id == _as_uuid(project_id))
            .order_by(SkillRow.confidence.desc())
            .limit(20)
        )
    elif session_id is not None:
        stmt = (
            select(SkillRow)
            .where(SkillRow.session_tag == session_id)
            .order_by(SkillRow.confidence.desc())
            .limit(20)
        )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    if not rows:
        return []

    if query_vec:
        q_tokens = set(question.lower().split())
        scored: list[tuple[float, SkillRow]] = []
        for row in rows:
            if row.text_embedding:
                score = _cosine(query_vec, row.text_embedding)
            else:
                r_tokens = set(row.text.lower().split())
                overlap = len(q_tokens & r_tokens)
                score = overlap / max(len(q_tokens | r_tokens), 1)
            scored.append((score, row))
        scored.sort(key=lambda kv: (-kv[0], -float(kv[1].confidence)))
        return [_row_to_skill(r) for s, r in scored if s > 0.1][:top_k]

    # 3. Pure token overlap — embedder unavailable.
    q_tokens = set(question.lower().split())
    overlap_scored = [
        (len(q_tokens & set(r.text.lower().split())), r) for r in rows
    ]
    overlap_scored.sort(key=lambda kv: (-kv[0], -float(kv[1].confidence)))
    return [_row_to_skill(r) for score, r in overlap_scored if score > 0][:top_k]


async def _vector_search_skills(
    query_vec: list[float],
    *,
    user_id: uuid.UUID | str | None,
    session_id: str | None,
    k: int,
    project_id: uuid.UUID | str | None = None,
):
    try:
        from app.core.config_loader import cfg
        from storage.vectors import get_vector_store

        store = get_vector_store()
        collection = _collection_for(user_id)
        await store.ensure_collection(
            collection,
            vector_size=len(query_vec),
            embedding_model=cfg.embeddings.model,
        )
        if project_id is not None:                  # §17: project scope wins
            flt = {"project_id": str(project_id)}
        elif session_id is not None:
            flt = {"session_tag": session_id}
        else:
            flt = None
        return await store.query(collection, vector=query_vec, k=k, filter=flt)
    except Exception as exc:
        log.debug("skill vector search fell through to JSONB: %s", exc)
        return []


async def _hits_to_skills(session: AsyncSession, hits) -> list[Skill]:
    point_ids = []
    for h in hits:
        try:
            point_ids.append(uuid.UUID(h.id))
        except ValueError:
            pass
    if not point_ids:
        return []
    result = await session.execute(
        select(SkillRow).where(SkillRow.vector_point_id.in_(point_ids))
    )
    by_pid = {r.vector_point_id: r for r in result.scalars().all()}
    out: list[Skill] = []
    for h in hits:
        try:
            pid = uuid.UUID(h.id)
        except ValueError:
            continue
        row = by_pid.get(pid)
        if row is not None:
            out.append(_row_to_skill(row))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def delete_skill(session: AsyncSession, skill_id: str) -> bool:
    try:
        sk_uuid = uuid.UUID(skill_id)
    except (TypeError, ValueError):
        return False
    row = await session.get(SkillRow, sk_uuid)
    if row is None:
        return False
    # Drop the vector too so the user-owned memory drawer's delete
    # actually erases the skill from search.
    if row.vector_point_id:
        try:
            from storage.vectors import get_vector_store

            store = get_vector_store()
            await store.delete(
                _collection_for(getattr(row, "user_id", None)),
                ids=[str(row.vector_point_id)],
            )
        except Exception as exc:
            log.warning("skill vector delete deferred: %s", exc)
    await session.delete(row)
    await session.commit()
    return True


def _row_to_skill(row: SkillRow) -> Skill:
    return Skill(
        id=str(row.id),
        session_id=row.session_tag,
        user_id=str(row.user_id) if row.user_id else None,
        project_id=str(row.project_id) if row.project_id else None,
        text=row.text,
        kind=row.kind,
        confidence=float(row.confidence),
        evidence_episode_ids=list(row.evidence_episode_ids or []),
        text_embedding=row.text_embedding,
        last_seen_ms=int(row.created_at.timestamp() * 1000) if row.created_at else 0,
    )
