"""Tests for the code knowledge graph core (app/codegraph).

Python extraction uses the stdlib `ast` (always available). The multi-language
tree-sitter tests skip when the grammar pack isn't installed.
"""
from __future__ import annotations

import pytest

from app.codegraph.builder import build_code_graph
from app.codegraph import query
from app.codegraph.summary import summarize_graph

PY = (
    "class Base:\n"
    "    pass\n"
    "\n"
    "class Foo(Base):\n"
    "    def bar(self, x):\n"
    "        return helper(x)\n"
    "\n"
    "def helper(x):\n"
    "    return x + 1\n"
)


def _ids(nodes):
    return {n["qualified_name"] for n in nodes}


# --- Python (ast) ---------------------------------------------------------

def test_python_nodes_and_kinds():
    g = build_code_graph([("pkg/service.py", PY)])
    names = {n.qualified_name: n.kind for n in g.nodes.values()}
    assert names["Base"] == "class"
    assert names["Foo"] == "class"
    assert names["Foo.bar"] == "method"
    assert names["helper"] == "function"
    assert "pkg/service.py" in g.nodes and g.nodes["pkg/service.py"].kind == "file"


def test_python_contains_edges():
    g = build_code_graph([("pkg/service.py", PY)])
    # file contains Foo; Foo contains bar
    assert any(e.src == "pkg/service.py" and e.dst.endswith("::Foo")
               for e in g.out_edges("pkg/service.py", "contains"))
    foo = "pkg/service.py::Foo"
    assert any(e.dst == "pkg/service.py::Foo.bar"
               for e in g.out_edges(foo, "contains"))


def test_python_call_edge_resolved():
    g = build_code_graph([("pkg/service.py", PY)])
    # Foo.bar calls helper
    edges = [e for e in g.edges if e.kind == "calls"]
    assert any(e.src == "pkg/service.py::Foo.bar" and e.dst == "pkg/service.py::helper"
               for e in edges)


def test_python_extends_edge_resolved():
    g = build_code_graph([("pkg/service.py", PY)])
    assert any(e.kind == "extends" and e.src == "pkg/service.py::Foo"
               and e.dst == "pkg/service.py::Base" for e in g.edges)


def test_python_import_edge_internal():
    files = [
        ("app/a.py", "from app.b import thing\n"),
        ("app/b.py", "def thing():\n    return 1\n"),
    ]
    g = build_code_graph(files)
    assert any(e.kind == "imports" and e.src == "app/a.py" and e.dst == "app/b.py"
               for e in g.edges)


# --- query tools ----------------------------------------------------------

def test_query_find_callers_callees():
    g = build_code_graph([("pkg/service.py", PY)])
    assert _ids(query.find_symbol(g, "helper")) == {"helper"}
    callers = query.callers(g, "pkg/service.py::helper")
    assert any(c["qualified_name"] == "Foo.bar" for c in callers)
    callees = query.callees(g, "pkg/service.py::Foo.bar")
    assert any(c["qualified_name"] == "helper" for c in callees)


def test_file_structure():
    g = build_code_graph([("pkg/service.py", PY)])
    fs = query.file_structure(g, "pkg/service.py")
    assert fs["exists"] and len(fs["symbols"]) >= 3


def test_summary_has_sections():
    g = build_code_graph([("pkg/service.py", PY)])
    s = summarize_graph(g)
    assert "Code knowledge graph" in s
    assert "pkg/service.py" in s
    assert "helper" in s


# --- tree-sitter (multi-language) ----------------------------------------

def test_treesitter_javascript():
    pytest.importorskip("tree_sitter_language_pack")
    js = (
        "class Widget {\n"
        "  render() { return draw(); }\n"
        "}\n"
        "function draw() { return 1; }\n"
    )
    g = build_code_graph([("ui/widget.js", js)])
    kinds = {n.name: n.kind for n in g.nodes.values() if n.kind != "file"}
    assert "Widget" in kinds and kinds["Widget"] == "class"
    assert "draw" in kinds
    # render() calls draw()
    assert any(e.kind == "calls" and g.nodes[e.dst].name == "draw"
               for e in g.edges if e.dst in g.nodes)


def test_treesitter_go():
    pytest.importorskip("tree_sitter_language_pack")
    go = (
        "package main\n"
        "func helper() int { return 1 }\n"
        "func main() { helper() }\n"
    )
    g = build_code_graph([("main.go", go)])
    names = {n.name for n in g.nodes.values() if n.kind == "function"}
    assert "helper" in names and "main" in names


# --- framework resolvers --------------------------------------------------

def test_framework_fastapi_routes():
    src = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        '@app.get("/users/{id}")\n'
        "async def get_user(id):\n"
        "    return {}\n"
        '@app.post("/users")\n'
        "def create_user():\n"
        "    return {}\n"
    )
    g = build_code_graph([("api/main.py", src)])
    routes = {n.qualified_name for n in g.nodes.values() if n.kind == "route"}
    assert {"GET /users/{id}", "POST /users"} <= routes
    refs = {(g.nodes[e.src].qualified_name, g.nodes[e.dst].qualified_name)
            for e in g.edges if e.kind == "references"}
    assert ("GET /users/{id}", "get_user") in refs
    assert ("POST /users", "create_user") in refs


def test_framework_django_urls():
    g = build_code_graph([
        ("app/urls.py",
         "from django.urls import path\nfrom . import views\n"
         'urlpatterns = [path("home/", views.home), path("about/", views.about)]\n'),
        ("app/views.py", "def home():\n    pass\ndef about():\n    pass\n"),
    ])
    refs = {(g.nodes[e.src].qualified_name, g.nodes[e.dst].qualified_name)
            for e in g.edges if e.kind == "references"}
    assert any(d == "home" for _, d in refs)


def test_framework_express_routes():
    pytest.importorskip("tree_sitter_language_pack")
    js = ('const router = require("express").Router();\n'
          'router.get("/items", listItems);\n'
          "function listItems(){ return 1; }\n")
    g = build_code_graph([("routes.js", js)])
    refs = {(g.nodes[e.src].qualified_name, g.nodes[e.dst].qualified_name)
            for e in g.edges if e.kind == "references"}
    assert ("GET /items", "listItems") in refs


# --- code-graph retrieval + tool registration -----------------------------

def test_evidence_from_graph():
    from app.codegraph.retrieval import evidence_from_graph
    g = build_code_graph([("pkg/service.py", PY)])
    ev = evidence_from_graph(g, "who calls helper and what does it do?")
    assert ev and any("helper" in e["text"] for e in ev)
    assert any("called by" in e["text"] for e in ev)


def test_import_aware_call_resolution():
    # Two files each define `process`; caller imports one explicitly → the call
    # edge must bind to the imported one, not fan out to both.
    files = [
        ("app/a.py", "from app.b import process\n"
                     "def run():\n    return process()\n"),
        ("app/b.py", "def process():\n    return 1\n"),
        ("app/c.py", "def process():\n    return 2\n"),
    ]
    g = build_code_graph(files)
    call_edges = [e for e in g.edges
                  if e.kind == "calls" and e.src == "app/a.py::run"]
    targets = {e.dst for e in call_edges}
    assert "app/b.py::process" in targets
    assert "app/c.py::process" not in targets   # not the wrong same-named one


def test_evidence_includes_overview():
    from app.codegraph.retrieval import evidence_from_graph, graph_document
    g = build_code_graph([("pkg/service.py", PY)])
    # An overview chunk is always present (so follow-ups keep project shape).
    ev = evidence_from_graph(g, "unrelated question with no symbol names")
    assert any(e["source"] == "code-graph:overview" for e in ev)
    # The embeddable graph document carries structure for RAG.
    doc = graph_document(g)
    assert "helper" in doc and "Symbols and relationships" in doc


def test_query_tools_registered():
    import app.codegraph.tools  # noqa: F401 — registers on import
    from app.tools import registry
    names = registry.names()
    for t in ("code_search", "code_callers", "code_callees", "code_impact",
              "code_file_structure"):
        assert t in names, t
