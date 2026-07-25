"""Tests for the universal preview matrix (vNext §8.8, Stage 8 Component C)."""
from __future__ import annotations

import asyncio
import io
import zipfile

import app.documents.preview as P


def _run(coro):
    return asyncio.run(coro)


# ---- descriptor routing ---------------------------------------------------
def test_raster_formats():
    for f in ("pdf", "docx", "pptx"):
        d = P.preview_descriptor(f)
        assert d.mode == P.RASTER and d.needs_rasterize and d.page_1_first


def test_sheet_formats():
    for f in ("xlsx", "csv"):
        assert P.preview_descriptor(f).mode == P.SHEETS


def test_sandbox_formats():
    for f in ("html", "svg", "react", "jsx"):
        d = P.preview_descriptor(f)
        assert d.mode == P.SANDBOX and d.needs_sandbox


def test_archive_and_json_and_text():
    assert P.preview_descriptor("zip").mode == P.MEMBER_TREE
    assert P.preview_descriptor("7z").mode == P.MEMBER_TREE
    assert P.preview_descriptor("json").mode == P.JSON_TREE
    assert P.preview_descriptor("md").mode == P.TEXT


def test_unknown_code_ext_is_text_binary_is_unsupported():
    assert P.preview_descriptor("py").mode == P.TEXT      # code-ish → text
    assert P.preview_descriptor("some/weird thing").mode == P.UNSUPPORTED
    assert P.preview_descriptor(".PDF").mode == P.RASTER  # normalized


def test_descriptor_never_raises():
    assert P.preview_descriptor(None).mode in (P.UNSUPPORTED, P.TEXT)  # type: ignore[arg-type]


def test_cache_key_stable_and_scoped():
    k1 = P.preview_cache_key("A1", 3, "csv")
    assert k1 == "A1:3:csv:0"
    assert P.preview_cache_key("A1", 3, "csv") == k1
    assert P.preview_cache_key("A1", 4, "csv") != k1   # version-scoped


# ---- pure builders --------------------------------------------------------
def test_sheets_from_csv_infers_types():
    s = P.sheets_from_csv("name,age,active\nAlice,30,true\nBob,25,false")[0]
    assert s.columns == ["name", "age", "active"]
    assert s.types == ["str", "int", "bool"]
    assert len(s.rows) == 2


def test_sheets_from_csv_float_and_truncation():
    body = "\n".join(f"{i},{i}.5" for i in range(600))
    s = P.sheets_from_csv("x,y\n" + body, max_rows=100)[0]
    assert s.types == ["int", "float"]
    assert s.truncated and len(s.rows) == 100


def test_sheets_from_csv_empty():
    assert P.sheets_from_csv("")[0].columns == []


def test_member_tree_nests():
    tree = P.member_tree(["src/main.py", "src/util/helm.py", "README.md"])
    assert tree == {"src": {"main.py": None, "util": {"helm.py": None}},
                    "README.md": None}


def test_member_tree_from_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a/b.txt", "x")
        z.writestr("c.md", "y")
    tree = P.member_tree_from_zip(buf.getvalue())
    assert tree == {"a": {"b.txt": None}, "c.md": None}


def test_member_tree_from_bad_bytes_is_empty():
    assert P.member_tree_from_zip(b"not a zip") == {}


# ---- build_preview dispatch ----------------------------------------------
def test_build_text(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("md", text="# Hi", artifact_id="A1", version=1))
    assert r.ok and r.mode == P.TEXT and r.payload["text"] == "# Hi"


def test_build_json_parses(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("json", text='{"a":1}', artifact_id="A1", version=1))
    assert r.ok and r.payload["json"] == {"a": 1}


def test_build_json_bad_shows_raw(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("json", text="{bad", artifact_id="A1", version=1))
    assert r.ok and r.payload["json"] is None and r.payload["raw"] == "{bad"


def test_build_csv_sheets(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("csv", text="a,b\n1,2", artifact_id="A1", version=1))
    assert r.ok and r.mode == P.SHEETS and len(r.payload["sheets"]) == 1


def test_build_xlsx_uses_injected_sheet_fn(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)

    async def sheet_fn(data):
        return [P.Sheet("Sheet1", ["x"], [[1]], ["int"])]
    r = _run(P.build_preview("xlsx", data=b"...", artifact_id="A1", version=1,
                             sheet_fn=sheet_fn))
    assert r.ok and r.payload["sheets"][0]["name"] == "Sheet1"


def test_build_raster_injected_and_page1(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)

    async def raster(data, fmt):
        return ["<img1>", "<img2>", "<img3>"]
    r = _run(P.build_preview("pdf", data=b"%PDF", artifact_id="A1", version=1,
                             rasterize_fn=raster))
    assert r.ok and r.page_count == 3 and r.payload["page_1_first"] is True


def test_build_raster_without_rasterizer_fails(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("docx", data=b"x", artifact_id="A1", version=1))
    assert not r.ok and "no rasterizer" in r.error


def test_build_member_tree(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x/y.py", "z")
    r = _run(P.build_preview("zip", data=buf.getvalue(), artifact_id="A1", version=1))
    assert r.ok and r.mode == P.MEMBER_TREE and r.payload["tree"] == {"x": {"y.py": None}}


def test_build_sandbox(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("html", text="<h1>x</h1>", artifact_id="A1", version=1))
    assert r.ok and r.mode == P.SANDBOX and r.payload["csp"] is True


def test_build_cached(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    cache: dict = {}
    r1 = _run(P.build_preview("csv", text="a\n1", artifact_id="A1", version=1, cache=cache))
    r2 = _run(P.build_preview("csv", text="a\n1", artifact_id="A1", version=1, cache=cache))
    assert r1 is r2 and len(cache) == 1


def test_build_disabled(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: False)
    r = _run(P.build_preview("csv", text="a\n1", artifact_id="A1", version=1))
    assert not r.ok and r.error == "disabled"


def test_build_unsupported(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)
    r = _run(P.build_preview("some weird thing", text="x", artifact_id="A1", version=1))
    assert not r.ok and r.mode == P.UNSUPPORTED
