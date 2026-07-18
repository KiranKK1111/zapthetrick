"""Tests for the blob-preview routing decision (app/api/routes_blob._route).

The routing is the part that decides what opens inline vs. download — exactly the
behavior we changed (only Word/Excel download; everything else previewable). It's
pulled into a pure function so it's testable without a blob store. Skips cleanly
when the app's import chain (asyncpg, …) isn't installed.
"""
from __future__ import annotations

import pytest

_mod = pytest.importorskip("app.api.routes_blob")
_route = _mod._route


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"])
def test_images(ext):
    assert _route(ext)[0] == "image"


@pytest.mark.parametrize("ext", [".docx", ".doc", ".xls"])
def test_word_and_legacy_excel_are_download_only(ext):
    assert _route(ext)[0] == "download"


def test_xlsx_previews_in_the_grid():
    # .xlsx now parses to a spreadsheet-grid preview (rich-viewers #3); .xls
    # (old binary, openpyxl can't read it) stays download-only.
    assert _route(".xlsx")[0] == "spreadsheet"
    assert _route(".xls")[0] == "download"


def test_pdf():
    assert _route(".pdf")[0] == "pdf"


@pytest.mark.parametrize("ext,fmt", [
    (".txt", "text"), (".md", "markdown"), (".json", "json"), (".csv", "csv"),
    (".py", "code"), (".yml", "code"), (".ts", "code"), (".sql", "code"),
    (".ipynb", "json"),
])
def test_known_text_and_code(ext, fmt):
    assert _route(ext) == ("text", fmt)


@pytest.mark.parametrize("ext", [
    ".pptx", ".ppt", ".odt", ".ods", ".odp",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".lz4",
])
def test_powerpoint_odf_and_archives_extract(ext):
    assert _route(ext)[0] == "extract"


@pytest.mark.parametrize("ext", [".exe", ".dll", ".ttf", ".woff2", ".bin", ".xyzzy", ""])
def test_unknown_and_binaries_sniff(ext):
    assert _route(ext)[0] == "sniff"


def test_only_word_and_legacy_excel_download_nothing_else():
    # The core guarantee: of a representative spread, ONLY docx/doc/xls are
    # download; every other type routes to a previewable path (.xlsx included).
    download = {".docx", ".doc", ".xls"}
    for ext in [".pdf", ".pptx", ".odt", ".ods", ".odp", ".zip", ".7z",
                ".txt", ".py", ".json", ".csv", ".png", ".md", ".yml",
                ".xlsx", ".exe", ".unknown"]:
        assert _route(ext)[0] != "download", ext
    for ext in download:
        assert _route(ext)[0] == "download", ext


def test_xlsx_preview_tables_renders_markdown():
    # A round-trip: build a tiny .xlsx, render it to markdown pipe tables.
    openpyxl = pytest.importorskip("openpyxl")
    import io

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "score"])
    ws.append(["Ada", 99])
    ws.append(["Alan", 87])
    buf = io.BytesIO()
    wb.save(buf)

    md = _mod._xlsx_preview_tables(buf.getvalue())
    assert "| name | score |" in md
    assert "| Ada | 99 |" in md
    assert "| Alan | 87 |" in md
