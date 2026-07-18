"""Phase 5 persistence — versioned artifact store.

The DB round-trip has no test harness (route tests mount DB-less apps), so this
pins the pure version logic, the fail-open contract, and the ORM model shape.
The live round-trip is verified by click-through.
"""
from __future__ import annotations

import asyncio
import uuid

from app.documents.store import next_version, record_generation


class TestVersionLogic:
    def test_first_version_is_one(self):
        assert next_version(None) == 1
        assert next_version(0) == 1

    def test_increments_past_max(self):
        assert next_version(1) == 2
        assert next_version(7) == 8


class TestFailOpen:
    def test_record_generation_never_raises_without_db(self, monkeypatch):
        # No session factory (DB not ready) → returns None, no exception.
        import app.documents.store as store
        monkeypatch.setattr(
            "storage.db.get_session_factory", lambda: None, raising=False)
        key = asyncio.run(record_generation(
            str(uuid.uuid4()), "# Doc\n\nbody", fmt="pdf"))
        assert key is None

    def test_record_generation_ignores_empty_content(self):
        key = asyncio.run(record_generation(str(uuid.uuid4()), "   "))
        assert key is None

    def test_record_generation_ignores_bad_session_id(self):
        key = asyncio.run(record_generation("not-a-uuid", "# Doc\n\nbody"))
        assert key is None


class TestModelShape:
    def test_columns_and_registration(self):
        from storage.models import Base, GeneratedDocument
        cols = {c.name for c in GeneratedDocument.__table__.columns}
        assert {"id", "session_id", "doc_key", "version", "title",
                "doc_format", "goal", "content_md", "meta", "created_at"} <= cols
        assert "generated_documents" in Base.metadata.tables

    def test_migration_chains_from_head(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "m0019",
            "storage/migrations/versions/0019_generated_documents.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert m.revision == "0019_generated_documents"
        assert m.down_revision == "0018_rename_vector_point_id"
