"""Phase-4 (ArchitectureVerdict.md): generated-artifact validation + repair +
degrade. Uses the REAL document generators (fpdf/python-docx/pptx/openpyxl are
bundled deps) — every produced artifact must validate structurally.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from app.documents import validators as V
from app.documents.generators import render_document

_CONTENT = "# Report\n\nHello **world**.\n\n- one\n- two\n"


class TestValidators:
    @pytest.mark.parametrize("fmt", ["pdf", "docx", "pptx", "xlsx", "zip",
                                     "md", "txt", "csv", "json"])
    def test_real_renders_validate(self, fmt):
        data, _mime, _ext = render_document(_CONTENT, fmt, title="T")
        v = V.validate_artifact(data, fmt)
        assert v.ok, f"{fmt}: {v.reason} ({v.method})"

    def test_empty_bytes_fail(self):
        assert not V.validate_artifact(b"", "pdf").ok

    def test_garbage_pdf_fails(self):
        v = V.validate_artifact(b"this is not a pdf at all", "pdf")
        assert not v.ok

    def test_truncated_docx_fails(self):
        data, _m, _e = render_document(_CONTENT, "docx", title="T")
        v = V.validate_artifact(data[: len(data) // 2], "docx")
        assert not v.ok

    def test_zip_missing_openxml_marker_fails_docx(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("whatever.txt", "hi")
        v = V.validate_artifact(buf.getvalue(), "docx")
        assert not v.ok and "Content_Types" in v.reason

    def test_bad_json_fails_good_json_passes(self):
        assert not V.validate_artifact(b"{not json", "json").ok
        assert V.validate_artifact(b'{"a": 1}', "json").ok

    def test_unknown_format_skipped_ok(self):
        v = V.validate_artifact(b"\x00\x01", "cad")
        assert v.ok and v.method == "skipped"


class TestRenderValidated:
    def test_happy_path_meta(self):
        data, _m, ext, meta = V.render_validated(_CONTENT, "pdf", title="T")
        assert meta["validated"] is True
        assert meta["repaired"] is False
        assert meta["degraded_from"] is None
        assert ext == "pdf" and data.startswith(b"%PDF-")

    def test_disabled_flag_skips_validation(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.artifact_validation, "enabled", False)
        _d, _m, _e, meta = V.render_validated(_CONTENT, "pdf", title="T")
        assert meta["method"] == "disabled"

    def test_repair_path(self, monkeypatch):
        # First validation fails, the re-render validates → repaired=True.
        calls = {"n": 0}
        real = V.validate_artifact

        def flaky(data, fmt):
            calls["n"] += 1
            if calls["n"] == 1:
                return V.ValidationResult(False, fmt, "test", "injected")
            return real(data, fmt)

        monkeypatch.setattr(V, "validate_artifact", flaky)
        _d, _m, ext, meta = V.render_validated(_CONTENT, "pdf", title="T")
        assert meta["validated"] is True and meta["repaired"] is True
        assert ext == "pdf"

    def test_degrade_path(self, monkeypatch):
        # PDF permanently invalid → walks pdf→docx, which validates.
        real = V.validate_artifact

        def pdf_always_bad(data, fmt):
            if fmt == "pdf":
                return V.ValidationResult(False, "pdf", "test", "injected")
            return real(data, fmt)

        monkeypatch.setattr(V, "validate_artifact", pdf_always_bad)
        _d, _m, ext, meta = V.render_validated(_CONTENT, "pdf", title="T")
        assert meta["validated"] is True
        assert meta["degraded_from"] == "pdf"
        assert ext == "docx"

    def test_never_blocks_delivery_when_all_fail(self, monkeypatch):
        monkeypatch.setattr(
            V, "validate_artifact",
            lambda d, f: V.ValidationResult(False, f, "test", "injected"))
        data, _m, ext, meta = V.render_validated(_CONTENT, "pdf", title="T")
        assert data and ext == "pdf"           # original bytes still shipped
        assert meta["validated"] is False      # honestly reported
