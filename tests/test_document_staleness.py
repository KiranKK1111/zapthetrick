"""Phase 5 — multi-format staleness marking."""
from __future__ import annotations

from dataclasses import dataclass

from app.documents.staleness import compute_staleness


@dataclass
class _Row:
    version: int
    doc_format: str


class TestComputeStaleness:
    def test_empty(self):
        r = compute_staleness([])
        assert r == {"latest_version": 0, "formats": [], "stale_formats": [],
                     "any_stale": False}

    def test_older_format_is_stale(self):
        # pdf@v1, then the source moved to v2 as docx → the pdf is stale.
        rows = [_Row(1, "pdf"), _Row(2, "docx")]
        r = compute_staleness(rows)
        assert r["latest_version"] == 2
        assert r["stale_formats"] == ["pdf"]
        assert r["any_stale"] is True
        by = {f["format"]: f for f in r["formats"]}
        assert by["pdf"]["stale"] is True
        assert by["docx"]["stale"] is False

    def test_all_current_when_reexported(self):
        # Both formats exist at the latest version → nothing stale.
        rows = [_Row(1, "pdf"), _Row(2, "docx"), _Row(2, "pdf")]
        r = compute_staleness(rows)
        assert r["any_stale"] is False
        assert r["stale_formats"] == []

    def test_accepts_tuples_and_dicts(self):
        r1 = compute_staleness([(1, "pdf"), (2, "docx")])
        r2 = compute_staleness([{"version": 1, "format": "pdf"},
                                {"version": 2, "doc_format": "docx"}])
        assert r1["stale_formats"] == ["pdf"] == r2["stale_formats"]

    def test_ignores_unparseable_rows(self):
        rows = [_Row(1, "pdf"), _Row("bad", "docx"), _Row(2, "docx")]
        r = compute_staleness(rows)
        assert r["latest_version"] == 2
        assert "pdf" in r["stale_formats"]


class TestStalenessEndpoint:
    def _client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from app.api.routes_documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_endpoint_fails_open_without_store(self, monkeypatch):
        import app.api.routes_documents  # noqa: F401
        # No DB configured in unit env → fail-open, non-stale report.
        c = self._client()
        r = c.get("/api/documents/artifacts/abc-123/staleness")
        assert r.status_code == 200
        body = r.json()
        assert body["any_stale"] is False
        assert body["stale_formats"] == []
