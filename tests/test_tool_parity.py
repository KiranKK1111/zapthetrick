"""P2-9 — tool parity finish: MultiEdit, permission-mode alignment, in-process MCP.

Pure/offline: the atomic multi_edit tool, Claude mode-name normalization +
plan-mode gating, and in-process MCP tool registration + invocation.
"""
from __future__ import annotations

import asyncio

from app.agent import permissions, tools


# ── MultiEdit ──────────────────────────────────────────────────────────────
def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(tmp_path)


def test_multi_edit_applies_all_atomically(tmp_path):
    ws = _write(tmp_path, "a.py", "x = 1\ny = 2\nz = 3\n")
    res = asyncio.run(tools.multi_edit(ws, path="a.py", edits=[
        {"old": "x = 1", "new": "x = 10"},
        {"old": "z = 3", "new": "z = 30"},
    ]))
    assert "applied 2 edit(s)" in res
    assert (tmp_path / "a.py").read_text() == "x = 10\ny = 2\nz = 30\n"


def test_multi_edit_is_all_or_nothing(tmp_path):
    ws = _write(tmp_path, "a.py", "x = 1\ny = 2\n")
    res = asyncio.run(tools.multi_edit(ws, path="a.py", edits=[
        {"old": "x = 1", "new": "x = 10"},
        {"old": "NOPE", "new": "q = 9"},     # this one fails
    ]))
    assert res.startswith("ERROR") and "edit #2" in res
    # nothing was written — the first edit must NOT have been persisted
    assert (tmp_path / "a.py").read_text() == "x = 1\ny = 2\n"


def test_multi_edit_rejects_ambiguous(tmp_path):
    ws = _write(tmp_path, "a.py", "v = 1\nv = 1\n")
    res = asyncio.run(tools.multi_edit(ws, path="a.py", edits=[
        {"old": "v = 1", "new": "v = 2"}]))
    assert "matches 2 times" in res
    assert (tmp_path / "a.py").read_text() == "v = 1\nv = 1\n"


def test_multi_edit_sequential_dependency(tmp_path):
    # edit 2's `old` only exists AFTER edit 1 ran → proves sequential application
    ws = _write(tmp_path, "a.py", "alpha\n")
    res = asyncio.run(tools.multi_edit(ws, path="a.py", edits=[
        {"old": "alpha", "new": "beta"},
        {"old": "beta", "new": "gamma"},
    ]))
    assert "applied 2 edit(s)" in res
    assert (tmp_path / "a.py").read_text() == "gamma\n"


def test_multi_edit_missing_file_and_bad_args(tmp_path):
    assert asyncio.run(tools.multi_edit(str(tmp_path), path="none.py",
                                        edits=[{"old": "a", "new": "b"}])
                       ).startswith("ERROR")
    ws = _write(tmp_path, "a.py", "x\n")
    assert asyncio.run(tools.multi_edit(ws, path="a.py", edits=[])
                       ).startswith("ERROR")


def test_multi_edit_registered():
    assert "multi_edit" in tools.HANDLERS
    assert "multi_edit" in tools.SPEC_BY_NAME
    assert tools.SPEC_BY_NAME["multi_edit"].writes is True


# ── permission-mode alignment ─────────────────────────────────────────────
def test_normalize_mode_accepts_claude_names():
    assert permissions.normalize_mode("default") == "ask"
    assert permissions.normalize_mode("bypassPermissions") == "auto"
    assert permissions.normalize_mode("plan") == "plan"
    assert permissions.normalize_mode("acceptEdits") == "acceptEdits"
    assert permissions.normalize_mode("nonsense") == "acceptEdits"  # fallback
    assert permissions.normalize_mode("") == "acceptEdits"


def test_decide_honors_claude_mode_names():
    # bypassPermissions → auto → everything allowed (deny-list aside)
    d, _ = permissions.decide("write", {"path": "x"}, "bypassPermissions")
    assert d == "allow"
    # default → ask → prompts even for a read
    d2, _ = permissions.decide("read", {"path": "x"}, "default")
    assert d2 == "ask"
    # plan is read-only → a write is denied
    d3, _ = permissions.decide("multi_edit", {"path": "x", "edits": []}, "plan")
    assert d3 == "deny"


# ── in-process MCP tools ─────────────────────────────────────────────────────
def test_in_process_register_list_and_invoke():
    from app import mcp
    from app.mcp import invoke
    from app.mcp.registry import registry

    mcp.reset_in_process()
    try:
        calls = {}

        def adder(args):
            calls["args"] = args
            return {"sum": args["a"] + args["b"]}

        mcp.register_in_process(
            "py_add", adder, description="add two numbers",
            input_schema={"a": "int", "b": "int"})

        # surfaced through the normal registry tool list
        names = [t.name for t in registry.list_tools()]
        assert "py_add" in names
        assert mcp.is_in_process("py_add")

        # invokable through the standard dispatcher
        out = asyncio.run(invoke("py_add", {"a": 2, "b": 5}))
        assert out == {"sum": 7}
        assert calls["args"] == {"a": 2, "b": 5}
    finally:
        mcp.reset_in_process()


def test_in_process_supports_async_and_unregister():
    from app import mcp
    from app.mcp import invoke
    from app.mcp.registry import registry

    mcp.reset_in_process()
    try:
        async def aecho(args):
            return {"echo": args.get("msg", "")}

        mcp.register_in_process("py_echo", aecho)
        out = asyncio.run(invoke("py_echo", {"msg": "hi"}))
        assert out == {"echo": "hi"}

        mcp.unregister_in_process("py_echo")
        assert not mcp.is_in_process("py_echo")
        assert "py_echo" not in [t.name for t in registry.list_tools()]
    finally:
        mcp.reset_in_process()


def test_in_process_non_dict_result_wrapped():
    from app import mcp
    from app.mcp import invoke

    mcp.reset_in_process()
    try:
        mcp.register_in_process("py_str", lambda args: "plain text")
        out = asyncio.run(invoke("py_str", {}))
        assert out == {"result": "plain text"}
    finally:
        mcp.reset_in_process()
