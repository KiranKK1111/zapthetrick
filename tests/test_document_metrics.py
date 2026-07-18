"""Phase 8 (subset) — document-generation metrics counter."""
from __future__ import annotations

from app.documents.metrics import (
    record_export, record_regeneration, record_section_edit, record_template,
    reset, snapshot,
)


def setup_function():
    reset()


def test_counts_and_by_format():
    record_export("pdf", ok=True)
    record_export("pdf", ok=True)
    record_export("docx", ok=True)
    snap = snapshot()
    assert snap["exports"] == 3
    assert snap["by_format"] == {"pdf": 2, "docx": 1}
    assert snap["failures"] == 0


def test_failures_and_latency():
    record_export("pdf", ok=True, latency_ms=100)
    record_export("pdf", ok=False, latency_ms=300)
    snap = snapshot()
    assert snap["exports"] == 2 and snap["failures"] == 1
    assert snap["avg_latency_ms"] == 200.0     # (100+300)/2


def test_empty_snapshot():
    reset()
    assert snapshot() == {
        "exports": 0, "failures": 0, "by_format": {}, "avg_latency_ms": 0.0,
        "regenerations": 0, "regeneration_rate": 0.0,
        "most_edited_sections": [], "template_success": {}}


def test_record_never_raises():
    # Junk args must not blow up the caller.
    record_export(None, ok=True)
    record_export("", ok=False)
    assert snapshot()["exports"] == 2


def test_untimed_export_does_not_drag_the_average_down():
    # The average is over the exports that TIMED themselves, so a caller that
    # records without a latency can't halve the mean.
    record_export("pdf", ok=True, latency_ms=200)
    record_export("md", ok=True)                 # no latency reported
    assert snapshot()["avg_latency_ms"] == 200.0


class TestExportEndpointRecordsRealLatency:
    """BUG 1 — the /export call sites never passed `latency_ms`, so
    `avg_latency_ms` was permanently 0. It must now be a real measurement."""

    def setup_method(self):
        reset()

    def _client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from app.api.routes_documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_successful_export_records_latency(self):
        c = self._client()
        r = c.post("/api/documents/export",
                   json={"content": "# Title\n\nHello", "format": "md",
                         "filename": "t"})
        assert r.status_code == 200
        snap = c.get("/api/documents/metrics").json()
        assert snap["exports"] == 1 and snap["by_format"] == {"md": 1}
        assert snap["avg_latency_ms"] > 0.0        # was always 0.0 before

    def test_failed_export_records_latency(self, monkeypatch):
        import app.api.routes_documents as routes
        import app.documents.validators as validators

        def _boom(*a, **kw):
            raise RuntimeError("render exploded")

        monkeypatch.setattr(validators, "render_validated", _boom)
        monkeypatch.setattr(routes, "render_document", _boom)   # legacy fallback
        c = self._client()
        r = c.post("/api/documents/export",
                   json={"content": "# T\n\nx", "format": "md"})
        assert r.status_code == 422
        snap = c.get("/api/documents/metrics").json()
        assert snap["failures"] == 1
        assert snap["avg_latency_ms"] > 0.0


class TestRoadmapSignals:
    """The three named cross-cutting metrics: regeneration rate, most-edited
    sections, template success."""

    def setup_method(self):
        reset()

    def test_regeneration_rate(self):
        record_export("pdf", ok=True)          # a fresh export
        record_regeneration()                  # one update
        snap = snapshot()
        assert snap["regenerations"] == 1
        # 1 regen / (1 export + 1 regen)
        assert snap["regeneration_rate"] == 0.5

    def test_most_edited_sections_ranked(self):
        record_section_edit("Database")
        record_section_edit("Database")
        record_section_edit("Security")
        record_section_edit("")                 # ignored
        top = snapshot()["most_edited_sections"]
        assert top[0] == {"section": "Database", "edits": 2}
        assert {"section": "Security", "edits": 1} in top

    def test_template_success_rate(self):
        record_template("modern", ok=True)
        record_template("modern", ok=True)
        record_template("modern", ok=False)
        record_template("", ok=True)            # ignored
        ts = snapshot()["template_success"]
        assert ts["modern"]["ok"] == 2 and ts["modern"]["fail"] == 1
        assert ts["modern"]["success_rate"] == 0.667

    def test_signal_recorders_never_raise(self):
        record_regeneration()
        record_section_edit(None)
        record_template(None)
        assert snapshot()["regenerations"] == 1


class TestExportRecordsTemplateSuccess:
    """A design-template export feeds the template-success signal."""

    def setup_method(self):
        reset()

    def _client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from app.api.routes_documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_template_export_records_success(self):
        c = self._client()
        resume = ("# Jane Doe\n\nemail@x.com\n\n## Summary\n\nEngineer.\n\n"
                  "## Skills\n\nPython, Go\n\n## Experience\n\n"
                  "**Dev · ACME** — 2020\n- Built things\n")
        r = c.post("/api/documents/export",
                   json={"content": resume, "format": "md",
                         "template": "modern", "filename": "cv"})
        assert r.status_code == 200
        ts = c.get("/api/documents/metrics").json()["template_success"]
        assert ts.get("modern", {}).get("ok") == 1
