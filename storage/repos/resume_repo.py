"""Resumes + resume chunks. The chunk side is the bulk of RAG ingest."""
from __future__ import annotations

import uuid

from sqlalchemy import delete, select, update

from ..models import Resume, ResumeChunk
from .base import Repo


class ResumeRepo(Repo):
    async def create(
        self,
        *,
        filename: str,
        file_path: str,
        display_name: str,
        profile: dict,
        raw_text: str,
        embedding_model: str,
        user_id: uuid.UUID | None = None,
        active: bool = False,
    ) -> Resume:
        row = Resume(
            user_id=user_id,
            filename=filename,
            file_path=file_path,
            display_name=display_name,
            profile=profile,
            raw_text=raw_text,
            embedding_model=embedding_model,
            active=active,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, resume_id: uuid.UUID | str) -> Resume | None:
        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        return await self.session.get(Resume, resume_id)

    async def list(self, *, user_id: uuid.UUID | None = None) -> list[Resume]:
        stmt = select(Resume).order_by(Resume.uploaded_at.desc())
        if user_id is not None:
            stmt = stmt.where(Resume.user_id == user_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_active(self, resume_id: uuid.UUID | str) -> None:
        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        row = await self.session.get(Resume, resume_id)
        if row is None:
            return
        # Single-active invariant per user.
        if row.user_id is not None:
            await self.session.execute(
                update(Resume)
                .where(Resume.user_id == row.user_id, Resume.id != resume_id)
                .values(active=False)
            )
        row.active = True

    async def delete(self, resume_id: uuid.UUID | str) -> None:
        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        await self.session.execute(delete(Resume).where(Resume.id == resume_id))

    # ---- Chunks -----------------------------------------------------
    async def replace_chunks(
        self,
        resume_id: uuid.UUID | str,
        chunks: list[dict],
    ) -> list[ResumeChunk]:
        """Delete prior chunks and bulk-insert a fresh set. Used by RAG ingest.

        Each dict carries: content, level, section_type, parent_id,
        summary, entity_tags, vector_point_id, position.
        """
        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        await self.session.execute(
            delete(ResumeChunk).where(ResumeChunk.resume_id == resume_id)
        )
        rows = [
            ResumeChunk(
                resume_id=resume_id,
                parent_id=c.get("parent_id"),
                level=c.get("level", 1),
                section_type=c.get("section_type"),
                content=c["content"],
                summary=c.get("summary"),
                entity_tags=c.get("entity_tags"),
                vector_point_id=c.get("vector_point_id"),
                position=c.get("position", 0),
            )
            for c in chunks
        ]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def fetch_chunks(
        self, resume_id: uuid.UUID | str
    ) -> list[ResumeChunk]:
        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        result = await self.session.execute(
            select(ResumeChunk)
            .where(ResumeChunk.resume_id == resume_id)
            .order_by(ResumeChunk.position)
        )
        return list(result.scalars().all())

    async def search_chunks_fts(
        self,
        resume_id: uuid.UUID | str,
        query: str,
        *,
        limit: int = 30,
    ) -> list[ResumeChunk]:
        """BM25-flavored keyword search via Postgres tsvector + GIN."""
        from sqlalchemy import func

        if isinstance(resume_id, str):
            resume_id = uuid.UUID(resume_id)
        ts_query = func.plainto_tsquery("english", query)
        rank = func.ts_rank(ResumeChunk.content_tsv, ts_query)
        stmt = (
            select(ResumeChunk)
            .where(
                ResumeChunk.resume_id == resume_id,
                ResumeChunk.content_tsv.op("@@")(ts_query),
            )
            .order_by(rank.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
