"""Tests that the preview text-extraction path actually works for the formats we
now preview (PowerPoint / OpenDocument / archives), using synthetic, dependency-
free containers (all are ZIPs / stdlib). Skips when the parser's import chain
(pdfplumber, …) isn't installed.
"""
from __future__ import annotations

import io
import zipfile

import pytest

_parser = pytest.importorskip("app.documents.parser")
extract_document_text = _parser.extract_document_text


def _zip(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_pptx_text_extracted():
    data = _zip({"ppt/slides/slide1.xml":
                 "<p><a:t>Hello Slide One</a:t><a:t>Bullet two</a:t></p>"})
    out = extract_document_text(data, "deck.pptx")
    assert "Hello Slide One" in out and "Bullet two" in out


def test_odt_text_extracted():
    data = _zip({"content.xml": "<text:p>Open document body text</text:p>"})
    out = extract_document_text(data, "doc.odt")
    assert "Open document body text" in out


def test_zip_archive_lists_and_extracts_members():
    data = _zip({"readme.txt": "archived file contents here",
                 "notes.md": "# heading\nsome notes"})
    out = extract_document_text(data, "bundle.zip")
    # Member listing AND member text are both present.
    assert "readme.txt" in out and "archived file contents here" in out
    assert "notes.md" in out and "some notes" in out
