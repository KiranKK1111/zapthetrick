"""Symbol search (code-intelligence R4, task 5.2).

Pins Property 4: definitions + usages from the index/graph, workspace scope, and
empty (→ caller text/vector fallback) when not found.
"""
from __future__ import annotations

from app.codeintel.index import build_index_from_files
from app.codeintel.search import find

_B = "def foo(x):\n    return x + 1\n"
_A = "from b import foo\n\ndef run():\n    return foo(41)\n"


def _index():
    return build_index_from_files([("b.py", _B), ("a.py", _A)])


def test_finds_definition():
    res = find(_index(), "foo")
    assert res["found"]
    def_files = {n.path for n in res["definitions"]}
    assert "b.py" in def_files


def test_finds_usages():
    res = find(_index(), "foo")
    use_names = {n.name for n in res["usages"]}
    assert "run" in use_names or len(res["usages"]) >= 1


def test_not_found_returns_empty_for_fallback():
    res = find(_index(), "NoSuchSymbol")
    assert res["found"] is False
    assert res["definitions"] == [] and res["usages"] == []


def test_blank_symbol_safe():
    res = find(_index(), "")
    assert res["found"] is False
