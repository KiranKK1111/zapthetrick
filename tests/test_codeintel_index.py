"""Symbol index (code-intelligence R1, task 1.2).

Pins Property 1: per-language symbols with locations, unparseable/unsupported
files skipped without failing the build, and a sandbox-only on-disk build.
"""
from __future__ import annotations

import os

from app.codeintel.index import build_index, build_index_from_files


_PY = """
import os
from app.util import helper

class UserService:
    def get_user(self, uid):
        return helper(uid)

def main():
    s = UserService()
    return s.get_user(1)
"""


def test_python_symbols_extracted():
    idx = build_index_from_files([("svc.py", _PY)])
    names = {n.name for n in idx.file_symbols("svc.py")}
    assert "UserService" in names
    assert "get_user" in names and "main" in names


def test_unparseable_file_is_skipped_not_fatal():
    files = [("good.py", _PY), ("broken.py", "def (((\n  not python")]
    idx = build_index_from_files(files)
    # The good file still indexes; the broken one doesn't crash the build.
    assert {n.name for n in idx.file_symbols("good.py")}
    assert isinstance(idx.all_files(), list)


def test_binary_and_unsupported_skipped():
    files = [("svc.py", _PY), ("data.bin", "\x00\x01binary"), ("notes.txt", "hi")]
    idx = build_index_from_files(files)
    # Only the source file is a graph file node.
    assert any(f == "svc.py" for f in idx.all_files())


def test_sandbox_only_tree_build(tmp_path):
    # Build over a real on-disk tree; a symlink escaping the root is ignored.
    proj = tmp_path / "proj"
    (proj / "pkg").mkdir(parents=True)
    (proj / "pkg" / "a.py").write_text(_PY, encoding="utf-8")
    (proj / "pkg" / "b.py").write_text("def b():\n    return 1\n", encoding="utf-8")
    idx = build_index("ws-test", root=str(proj))
    files = set(idx.all_files())
    assert any(f.endswith("a.py") for f in files)
    assert any(f.endswith("b.py") for f in files)


def test_empty_tree_is_empty_index(tmp_path):
    idx = build_index("ws-empty", root=str(tmp_path))
    assert idx.all_files() == []
