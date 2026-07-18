"""Phase 5 (core) — semantic anchors, incremental section updates, version diff.

Pure model operations; persistence (versioned artifact store) is a later step."""
from __future__ import annotations

from app.documents.lifecycle import (
    anchor_for, diff_models, find_section, merge_update, remove_section,
    replace_section,
)
from app.documents.model import markdown_to_model, model_to_markdown

DOC = """# System Design

Intro.

## Overview

The overview.

## Database

Uses MySQL.

## Deployment

On Kubernetes.
"""


class TestAnchors:
    def test_slug(self):
        assert anchor_for("Security & Auth") == "security-auth"
        assert anchor_for("  Data Model  ") == "data-model"

    def test_find_by_title_and_anchor(self):
        m = markdown_to_model(DOC)
        assert find_section(m, "Database") is not None
        assert find_section(m, "database").heading == "Database"
        assert find_section(m, "Nonexistent") is None


class TestIncrementalUpdate:
    def test_replace_only_target_section(self):
        m = markdown_to_model(DOC)
        m2 = replace_section(m, "Database", "Now uses **PostgreSQL** with pgvector.")
        # The Database section changed...
        db = find_section(m2, "Database")
        assert "PostgreSQL" in model_to_markdown(
            type(m2)(sections=[db]))
        # ...and every OTHER section is untouched.
        assert "The overview." in model_to_markdown(
            type(m2)(sections=[find_section(m2, "Overview")]))
        assert "Kubernetes" in model_to_markdown(
            type(m2)(sections=[find_section(m2, "Deployment")]))
        # Same number of sections (replace, not append).
        assert len(m2.sections) == len(m.sections)

    def test_update_missing_section_appends(self):
        m = markdown_to_model(DOC)
        m2 = replace_section(m, "Security", "Uses JWT + OAuth2.")
        assert find_section(m2, "Security") is not None
        assert len(m2.sections) == len(m.sections) + 1

    def test_remove_section(self):
        m = markdown_to_model(DOC)
        m2 = remove_section(m, "Deployment")
        assert find_section(m2, "Deployment") is None
        assert len(m2.sections) == len(m.sections) - 1

    def test_original_unmutated(self):
        m = markdown_to_model(DOC)
        n = len(m.sections)
        replace_section(m, "Database", "changed")
        assert len(m.sections) == n and find_section(m, "Database") is not None


class TestMergeUpdate:
    """UPDATE_EXISTING core: fold this turn's edit into the prior document."""

    def test_add_new_section_appends(self):
        merged = merge_update(DOC, "Adding it now:\n\n## Caching\n\nUses Redis.")
        assert "Caching" in merged and "Redis" in merged
        assert "MySQL" in merged and "Kubernetes" in merged   # rest preserved

    def test_update_existing_section_replaces_body(self):
        merged = merge_update(DOC, "## Database\n\nNow uses PostgreSQL + pgvector.")
        assert "PostgreSQL" in merged and "Uses MySQL." not in merged  # old body gone
        assert "The overview." in merged                       # others untouched

    def test_no_headed_update_returns_prior_unchanged(self):
        # Conversational reply with no ## section → nothing to merge.
        merged = merge_update(DOC, "Sure, I can help with that!")
        assert "Database" in merged and "Overview" in merged
        # Same section set as the original.
        from app.documents.model import markdown_to_model
        assert {h for _, h in markdown_to_model(merged).headings()} == \
               {h for _, h in markdown_to_model(DOC).headings()}

    def test_merge_records_regeneration_and_section_edits(self):
        # Cross-cutting metrics: an UPDATE is a regeneration + one edit/section.
        from app.documents.metrics import reset, snapshot
        reset()
        merge_update(DOC, "## Database\n\nNow PostgreSQL.\n\n## Caching\n\nRedis.")
        snap = snapshot()
        assert snap["regenerations"] == 1
        edited = {s["section"] for s in snap["most_edited_sections"]}
        assert "Database" in edited and "Caching" in edited

    def test_no_headed_update_records_nothing(self):
        from app.documents.metrics import reset, snapshot
        reset()
        merge_update(DOC, "Just a chat reply, no sections.")
        assert snapshot()["regenerations"] == 0


class TestDiff:
    def test_added_removed_changed(self):
        old = markdown_to_model(DOC)
        new = remove_section(
            replace_section(old, "Database", "Now PostgreSQL."), "Deployment")
        new = replace_section(new, "Security", "New section.")
        d = diff_models(old, new)
        assert "Security" in d.added
        assert "Deployment" in d.removed
        assert "Database" in d.changed
        assert "Overview" in d.unchanged
        assert not d.is_empty

    def test_identical_models_have_empty_diff(self):
        m = markdown_to_model(DOC)
        d = diff_models(m, markdown_to_model(DOC))
        assert d.is_empty and set(d.unchanged) >= {"Overview", "Database"}

    def test_as_dict(self):
        d = diff_models(markdown_to_model(DOC),
                        replace_section(markdown_to_model(DOC), "Database", "x"))
        js = d.as_dict()
        assert set(js) == {"added", "removed", "changed", "unchanged"}
        assert "Database" in js["changed"]


# ── BUG 3: the diff engine is now reachable over REST ────────────────────────
V2 = DOC.replace("Uses MySQL.", "Uses PostgreSQL.") + "\n## Security\n\nJWT.\n"


class _Row:
    """Stand-in for a GeneratedDocument row (no test DB harness — see
    tests/test_document_store.py)."""

    def __init__(self, version: int, content_md: str, title: str = "Design"):
        self.version = version
        self.content_md = content_md
        self.title = title
        self.doc_key = "k1"


class _Factory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _diff_client(monkeypatch, rows, *, no_store: bool = False):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from app.api.routes_documents import router

    async def _list_versions(_session, _doc_key):
        return rows

    monkeypatch.setattr("app.documents.store.list_versions", _list_versions)
    monkeypatch.setattr("storage.db.get_session_factory",
                        lambda: (None if no_store else _Factory()))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestDiffEndpoint:
    def test_explicit_versions(self, monkeypatch):
        c = _diff_client(monkeypatch, [_Row(1, DOC), _Row(2, V2)])
        r = c.get("/api/documents/artifacts/k1/diff", params={"from": 1, "to": 2})
        assert r.status_code == 200
        js = r.json()
        assert js["from_version"] == 1 and js["to_version"] == 2
        assert "Security" in js["diff"]["added"]
        assert "Database" in js["diff"]["changed"]
        assert "Overview" in js["diff"]["unchanged"]
        assert js["is_empty"] is False
        # Both sources ride along so the FE can also line-diff (diff_view.dart).
        assert js["from_content_md"] == DOC and js["to_content_md"] == V2

    def test_defaults_to_the_last_two_versions(self, monkeypatch):
        c = _diff_client(monkeypatch, [_Row(1, DOC), _Row(2, V2)])
        js = c.get("/api/documents/artifacts/k1/diff").json()
        assert js["from_version"] == 1 and js["to_version"] == 2
        assert "Database" in js["diff"]["changed"]

    def test_single_version_is_an_empty_diff(self, monkeypatch):
        c = _diff_client(monkeypatch, [_Row(1, DOC)])
        js = c.get("/api/documents/artifacts/k1/diff").json()
        assert js["from_version"] == 1 and js["to_version"] == 1
        assert js["is_empty"] is True

    def test_unknown_version_404(self, monkeypatch):
        c = _diff_client(monkeypatch, [_Row(1, DOC)])
        r = c.get("/api/documents/artifacts/k1/diff", params={"from": 1, "to": 9})
        assert r.status_code == 404

    def test_unknown_document_404(self, monkeypatch):
        c = _diff_client(monkeypatch, [])
        assert c.get("/api/documents/artifacts/nope/diff").status_code == 404

    def test_no_store_503(self, monkeypatch):
        # No DB → an explicit "unavailable", never a silent empty diff.
        c = _diff_client(monkeypatch, [_Row(1, DOC)], no_store=True)
        assert c.get("/api/documents/artifacts/k1/diff").status_code == 503
