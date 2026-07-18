"""Dependency + call graphs (code-intelligence R2/R3, task 3.2).

Pins Properties 2 & 3: file→file dependency edges with externals reported;
symbol→symbol call edges + best-effort usages.
"""
from __future__ import annotations

from app.codeintel.index import build_index_from_files
from app.codeintel.graph import (
    dependency_graph, call_graph, dependents_of, usages_of,
)

_B = """
def foo(x):
    return x + 1
"""

_A = """
import os
from b import foo

def run():
    return foo(41)
"""


def _index():
    return build_index_from_files([("b.py", _B), ("a.py", _A)])


def test_internal_dependency_edge():
    dg = dependency_graph(_index())
    # a.py imports b.py (internal).
    assert "b.py" in dg["internal"].get("a.py", [])


def test_external_imports_reported():
    dg = dependency_graph(_index())
    # `os` doesn't resolve to a workspace file → external for a.py.
    assert "os" in dg["external"].get("a.py", [])


def test_dependents_of():
    deps = dependents_of(_index(), "b.py")
    assert "a.py" in deps              # a.py depends on b.py


def test_call_graph_has_edges():
    cg = call_graph(_index())
    # run() → foo() should produce at least one call edge.
    assert any(cg.values())


def test_usages_of_symbol_best_effort():
    uses = usages_of(_index(), "foo")
    # run references foo → at least one referencing symbol returned.
    names = {n.name for n in uses}
    assert "run" in names or len(uses) >= 1


def test_no_imports_no_internal_edges():
    idx = build_index_from_files([("solo.py", "def only():\n    return 1\n")])
    dg = dependency_graph(idx)
    assert dg["internal"].get("solo.py", []) == []
