"""Tests for the mermaid backend render lane (vNext §5.4, Stage 8 Component B)."""
from __future__ import annotations

import asyncio

import app.response_arch.mermaid as M


def _run(coro):
    return asyncio.run(coro)


# ---- strip_fence / source_hash -------------------------------------------
def test_strip_fence_extracts_body():
    body = M.strip_fence("```mermaid\nflowchart TD\n A-->B\n```")
    assert body.startswith("flowchart TD")
    assert "```" not in body


def test_strip_fence_passthrough_when_unfenced():
    assert M.strip_fence("flowchart TD\n A-->B").startswith("flowchart TD")


def test_source_hash_stable_and_differs():
    assert M.source_hash("a") == M.source_hash("a")
    assert M.source_hash("a") != M.source_hash("b")


# ---- lint_and_normalize ---------------------------------------------------
def test_missing_header_assumed_flowchart():
    n = M.lint_and_normalize("A[x] --> B[y]")
    assert n.header == "flowchart"
    assert n.source.splitlines()[0] == "flowchart TD"
    assert n.changed
    assert any("assumed flowchart" in w for w in n.warnings)


def test_bare_graph_gets_direction():
    n = M.lint_and_normalize("graph\n A-->B")
    assert n.source.splitlines()[0] == "graph TD"
    assert n.changed


def test_existing_direction_preserved():
    n = M.lint_and_normalize("flowchart LR\n A-->B")
    assert n.source.splitlines()[0] == "flowchart LR"


def test_long_label_wraps():
    n = M.lint_and_normalize(
        "flowchart TD\n A[This is a very long label that must wrap somewhere] --> B[x]",
        wrap=20)
    assert "<br/>" in n.source
    assert n.changed


def test_special_chars_quoted():
    n = M.lint_and_normalize("flowchart LR\n A[call foo() : bar] --> B[ok]")
    line = next(l for l in n.source.splitlines() if "foo" in l)
    assert '"' in line                 # the special-char label is quoted


def test_sequence_diagram_gets_no_direction():
    n = M.lint_and_normalize("sequenceDiagram\n Alice->>Bob: Hi")
    assert n.header == "sequencediagram"
    assert n.source.splitlines()[0] == "sequenceDiagram"   # untouched, no TD


def test_runaway_graph_is_capped():
    big = "flowchart TD\n" + "\n".join(
        f"N{i}[a] --> N{i+1}[b]" for i in range(30))
    n = M.lint_and_normalize(big, max_nodes=10)
    assert n.capped
    assert any("cap" in w for w in n.warnings)


def test_empty_source():
    n = M.lint_and_normalize("")
    assert n.source == ""
    assert "empty source" in n.warnings


def test_normalize_never_raises():
    n = M.lint_and_normalize(None)      # type: ignore[arg-type]
    assert isinstance(n, M.NormalizedDiagram)


# ---- render (injected seams) ---------------------------------------------
def test_render_disabled_returns_normalized_no_svg(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: False)

    async def good(src):
        return "<svg/>"
    r = _run(M.render("A[x]-->B[y]", render_fn=good))
    assert r.ok
    assert r.svg == ""                  # fail-open: normalized but unrendered
    assert r.envelope()["svg_artifact_id"] is None


def test_render_success_and_cache(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)

    async def good(src):
        return "<svg>" + src[:4] + "</svg>"
    cache: dict = {}
    r = _run(M.render("A[x]-->B[y]", render_fn=good, cache=cache))
    assert r.ok and r.svg and not r.from_cache
    assert len(cache) == 1
    r2 = _run(M.render("A[x]-->B[y]", render_fn=good, cache=cache))
    assert r2.from_cache is True        # unchanged source served from cache


def test_render_error_then_repair(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)

    async def picky(src):
        if "fixed" not in src:
            raise RuntimeError("parse error")
        return "<svg>ok</svg>"

    async def repair(src, err):
        return "flowchart TD\n A[fixed]-->B[ok]"
    r = _run(M.render("A[bad]-->B[y]", render_fn=picky, repair_fn=repair))
    assert r.ok and r.repaired and r.svg == "<svg>ok</svg>"


def test_render_error_no_repair_is_fail_open(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)

    async def bad(src):
        raise RuntimeError("boom")
    r = _run(M.render("A[x]-->B[y]", render_fn=bad))
    assert r.ok                         # never raises out
    assert r.svg == ""
    assert "render failed" in r.error


def test_repair_also_fails_is_fail_open(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)

    async def bad(src):
        raise RuntimeError("boom")

    async def repair(src, err):
        return "still broken"
    r = _run(M.render("A[x]-->B[y]", render_fn=bad, repair_fn=repair))
    assert r.ok and r.svg == ""
    assert "repair failed" in r.error


def test_envelope_shape(monkeypatch):
    monkeypatch.setattr(M, "enabled", lambda: True)

    async def good(src):
        return "<svg/>"
    r = _run(M.render("A[x]-->B[y]", render_fn=good))
    env = r.envelope()
    assert set(env) == {"mermaid_source", "svg_artifact_id", "warnings", "repaired"}
    assert env["svg_artifact_id"] is not None
