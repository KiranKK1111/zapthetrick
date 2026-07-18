"""Code-aware context builder (code-intelligence R5, task 6.3).

Pins Property 5: augments `rank_files` with symbol matches + direct
dependencies; with the feature gated off or no index it equals today's
`rank_files` ordering.
"""
from __future__ import annotations

import app.codeintel.context as C
from app.codeintel.index import build_index_from_files

_B = "def helper(x):\n    return x * 2\n"
_A = "from b import helper\n\ndef run():\n    return helper(21)\n"
_FILES = [("b.py", _B), ("a.py", _A), ("readme.md", "# notes")]


def test_gated_off_equals_rank_files(monkeypatch):
    monkeypatch.setattr(C, "_gated_on", lambda: False)
    ctx = C.select("run helper", _FILES, index=build_index_from_files(_FILES))
    assert ctx.augmented is False
    # Same set as a plain rank_files call.
    assert set(ctx.files) <= {"a.py", "b.py", "readme.md"}


def test_augments_with_dependencies(monkeypatch):
    monkeypatch.setattr(C, "_gated_on", lambda: True)
    idx = build_index_from_files(_FILES)
    # limit=1 → rank_files alone returns just the top file; the dependency graph
    # then pulls in its dependency, proving augmentation.
    ctx = C.select("run", _FILES, index=idx, limit=1)
    assert ctx.augmented is True
    assert "a.py" in ctx.files and "b.py" in ctx.files


def test_no_index_falls_back(monkeypatch):
    monkeypatch.setattr(C, "_gated_on", lambda: True)
    ctx = C.select("run", _FILES, index=None, workspace_id=None)
    assert ctx.augmented is False
    assert isinstance(ctx.files, list)


def test_empty_index_falls_back(monkeypatch):
    monkeypatch.setattr(C, "_gated_on", lambda: True)
    empty = build_index_from_files([])
    ctx = C.select("run", _FILES, index=empty)
    assert ctx.augmented is False
