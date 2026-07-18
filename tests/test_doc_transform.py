"""Universal Document & Code Transformation flow (P4 #20) — orchestration."""
from __future__ import annotations

import asyncio

from app.documents.transform import classify, transform_content


def test_classify_by_extension_and_content():
    assert classify("print(1)", "main.py") == ("code", "py")
    assert classify("# Title\n\nbody", "notes.md")[0] == "markdown"
    assert classify("plain words here", "a.txt") == ("text", "txt")
    # No extension → sniff markdown structure
    assert classify("## Heading\n- item")[0] == "markdown"


def test_flow_runs_a_transformer_then_reports():
    async def upper(s: str) -> str:
        return s.upper()

    r = asyncio.run(transform_content("# hello\n\nworld", filename="x.md",
                                      transformer=upper))
    assert r.kind == "markdown"
    assert "HELLO" in r.content       # transformer applied
    assert r.validated is True        # non-empty doc → valid
    assert r.formatted is True        # polish ran


def test_flow_is_fail_open_without_transformer():
    r = asyncio.run(transform_content("just some text", filename="n.txt"))
    assert r.kind == "text" and r.validated is True
    d = r.as_dict()
    assert d["kind"] == "text" and d["chars"] > 0


def test_code_kind_detected_and_language_mapped():
    # Sandbox/formatter may be unavailable offline → fail-open, but kind+lang set.
    r = asyncio.run(transform_content("def f():\n    return 1\n", filename="f.py"))
    assert r.kind == "code" and r.language == "python"
