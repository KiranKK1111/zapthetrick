"""Incremental re-indexing (code-intelligence R6, task 8.2).

Pins Property 6: a changed file updates only that file's symbols; a deletion
removes it; an absent prior index → full re-index fallback.
"""
from __future__ import annotations

from app.codeintel.index import build_index_from_files, cache_clear
from app.codeintel.reindex import reindex_files, reindex

_A_V1 = "def run():\n    return 1\n"
_A_V2 = "def run():\n    return 2\n\ndef extra():\n    return 3\n"
_B = "def foo():\n    return 0\n"


def setup_function(_):
    cache_clear()


def test_changed_file_reflected():
    idx = build_index_from_files([("a.py", _A_V1), ("b.py", _B)])
    assert {n.name for n in idx.file_symbols("a.py")} == {"run"}
    idx2 = reindex_files(idx, {"a.py": _A_V2})
    names = {n.name for n in idx2.file_symbols("a.py")}
    assert "extra" in names                       # new symbol picked up
    # The untouched file is preserved.
    assert {n.name for n in idx2.file_symbols("b.py")} == {"foo"}


def test_deletion_removes_file():
    idx = build_index_from_files([("a.py", _A_V1), ("b.py", _B)])
    idx2 = reindex_files(idx, {"b.py": None})
    assert "b.py" not in set(idx2.all_files())
    assert "a.py" in set(idx2.all_files())


def test_full_reindex_fallback_when_no_prior(tmp_path):
    # No cached index for this workspace → reindex does a full build over root.
    (tmp_path / "x.py").write_text(_A_V1, encoding="utf-8")
    idx = reindex("ws-new", changed=["x.py"], root=str(tmp_path))
    assert any(f.endswith("x.py") for f in idx.all_files())


def test_reindex_files_never_raises_on_garbage():
    idx = build_index_from_files([("a.py", _A_V1)])
    idx2 = reindex_files(idx, {"a.py": "def (((broken"})
    # Broken source doesn't crash; index still usable.
    assert isinstance(idx2.all_files(), list)
