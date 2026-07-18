"""Phase 6 — AST semantic editing (#21).

Pure/offline edits over source strings via tree-sitter, plus the agent-tool
wrappers (read→apply→write). Requires the tree-sitter language pack (already a
dependency); a couple of cases skip gracefully if a language parser is absent.
"""
from __future__ import annotations

import asyncio

from app.codegraph.edit import add_method, insert_import, rename_symbol
from app.codegraph.tsutil import get_parser_for

_PY = "python"
_HAVE_PY = get_parser_for(_PY) is not None

PY_SRC = (
    "import os\n"
    "\n"
    "class Foo:\n"
    "    def bar(self):\n"
    "        return os.getcwd()\n"
    "\n"
    "def baz():\n"
    "    return Foo()\n"
)


# ── rename ──────────────────────────────────────────────────────────────────
def test_rename_renames_identifiers():
    r = rename_symbol(PY_SRC, "Foo", "Widget", path="m.py")
    assert r.ok and r.changed
    assert "class Widget:" in r.source
    assert "return Widget()" in r.source
    assert "Foo" not in r.source


def test_rename_skips_strings_and_comments():
    src = '# Foo is great\nx = "Foo here"\nFoo = 1\nprint(Foo)\n'
    r = rename_symbol(src, "Foo", "Bar", path="m.py")
    assert r.ok
    assert "# Foo is great" in r.source        # comment untouched
    assert '"Foo here"' in r.source            # string untouched
    assert "Bar = 1" in r.source and "print(Bar)" in r.source


def test_rename_missing_symbol_fails_cleanly():
    r = rename_symbol(PY_SRC, "Nope", "X", path="m.py")
    assert not r.ok and not r.changed
    assert "not found" in r.detail


def test_rename_identical_names_rejected():
    r = rename_symbol(PY_SRC, "Foo", "Foo", path="m.py")
    assert not r.ok


def test_rename_no_parser_fails_gracefully():
    r = rename_symbol("x=1", "x", "y", path="data.unknownext")
    assert not r.ok and "no tree-sitter parser" in r.detail


# ── insert import ─────────────────────────────────────────────────────────
def test_insert_import_after_existing():
    r = insert_import(PY_SRC, "import sys", path="m.py")
    assert r.ok
    lines = r.source.splitlines()
    assert "import os" in lines and "import sys" in lines
    # New import sits right after the existing import block, before the class.
    assert lines.index("import sys") < lines.index("class Foo:")


def test_insert_import_dedup():
    r = insert_import(PY_SRC, "import os", path="m.py")
    assert not r.ok and "already present" in r.detail


def test_insert_import_no_existing_imports():
    src = "def f():\n    return 1\n"
    r = insert_import(src, "import math", path="m.py")
    assert r.ok
    assert r.source.splitlines()[0] == "import math"


# ── add method ────────────────────────────────────────────────────────────
def test_add_method_python_indented():
    r = add_method(PY_SRC, "Foo", "def qux(self):\n    return 42", path="m.py")
    assert r.ok and r.changed
    assert "    def qux(self):" in r.source
    assert "        return 42" in r.source
    # Inserted inside the class, before the module-level baz().
    assert r.source.index("def qux") < r.source.index("def baz")


def test_add_method_missing_class_fails():
    r = add_method(PY_SRC, "Ghost", "def q(self): pass", path="m.py")
    assert not r.ok and "not found" in r.detail


def test_add_method_brace_language():
    if get_parser_for("javascript") is None:
        import pytest
        pytest.skip("javascript parser unavailable")
    js = "class A {\n  m() { return 1; }\n}\n"
    r = add_method(js, "A", "greet() { return 2; }", path="a.js")
    assert r.ok
    assert "greet()" in r.source
    # Inserted before the class's closing brace.
    assert r.source.rstrip().endswith("}")
    assert r.source.index("greet()") < r.source.rfind("}")


# ── agent-tool wrappers (read → apply → write) ─────────────────────────────
def test_tools_registered_and_gated():
    from app.agent import permissions
    from app.agent.tools import HANDLERS, SPEC_BY_NAME

    for name in ("rename_symbol", "insert_import", "add_method"):
        assert name in HANDLERS
        assert SPEC_BY_NAME[name].writes is True
        # write tools are denied in read-only plan mode, allowed in acceptEdits.
        assert permissions.decide(name, {}, "plan")[0] == "deny"
        assert permissions.decide(name, {}, "acceptEdits")[0] == "allow"


def test_rename_tool_applies_to_file(tmp_path):
    from app.agent.tools import HANDLERS

    f = tmp_path / "m.py"
    f.write_text(PY_SRC, encoding="utf-8")
    out = asyncio.run(HANDLERS["rename_symbol"](
        str(tmp_path), path="m.py", old="Foo", new="Gadget"))
    assert "renamed" in out
    assert "class Gadget:" in f.read_text(encoding="utf-8")


def test_add_method_tool_applies_to_file(tmp_path):
    from app.agent.tools import HANDLERS

    f = tmp_path / "m.py"
    f.write_text(PY_SRC, encoding="utf-8")
    out = asyncio.run(HANDLERS["add_method"](
        str(tmp_path), path="m.py", class_name="Foo",
        code="def qux(self):\n    return 1"))
    assert "added member" in out
    assert "def qux" in f.read_text(encoding="utf-8")


def test_ast_tool_missing_file_errors(tmp_path):
    from app.agent.tools import HANDLERS

    out = asyncio.run(HANDLERS["insert_import"](
        str(tmp_path), path="nope.py", import_line="import os"))
    assert "no such file" in out
