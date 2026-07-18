"""Render a CodeGraph into a compact, LLM-friendly project overview.

Goes into the answer context so the model understands the codebase's shape:
stats, a directory tree, each file's top classes/functions (with signatures),
the most-referenced symbols, and key internal import edges — all under a char
budget.
"""
from __future__ import annotations

from .model import CodeGraph

_DEF_KINDS_ORDER = {"class": 0, "interface": 0, "struct": 0, "trait": 0,
                    "enum": 0, "function": 1, "method": 2}


def _dir_tree(paths: list[str], max_lines: int = 40) -> str:
    """A compact directory listing (dirs with file counts), capped."""
    from collections import Counter
    dirs = Counter()
    for p in paths:
        d = p.rsplit("/", 1)[0] if "/" in p else "."
        dirs[d] += 1
    lines = [f"  {d}/  ({n} files)" for d, n in sorted(dirs.items())[:max_lines]]
    if len(dirs) > max_lines:
        lines.append(f"  … {len(dirs) - max_lines} more directories")
    return "\n".join(lines)


def summarize_graph(graph: CodeGraph, *, max_chars: int = 6000) -> str:
    st = graph.stats()
    langs = ", ".join(f"{k} ({v})" for k, v in
                      sorted(st["languages"].items(), key=lambda kv: -kv[1]))
    out: list[str] = []
    out.append(
        f"# Code knowledge graph\n"
        f"{st['files_parsed']} source files, {st['nodes']} symbols, "
        f"{st['edges']} relationships. Languages: {langs or 'n/a'}."
    )

    files = sorted({n.path for n in graph.files})
    out.append("\n## Structure\n" + _dir_tree(files))

    # Most-referenced callables (incoming 'calls' edges) — likely entry/core.
    indeg: dict[str, int] = {}
    for e in graph.edges:
        if e.kind == "calls":
            indeg[e.dst] = indeg.get(e.dst, 0) + 1
    top = sorted(indeg.items(), key=lambda kv: -kv[1])[:12]
    if top:
        out.append("\n## Most-referenced symbols")
        for nid, n in top:
            node = graph.nodes.get(nid)
            if node:
                out.append(f"- `{node.qualified_name}` ({node.path}) — {n} callers")

    # Per-file symbol outline (bounded).
    out.append("\n## Files & symbols")
    for path in files:
        syms = graph.nodes_in_file(path)
        if not syms:
            continue
        syms.sort(key=lambda n: (_DEF_KINDS_ORDER.get(n.kind, 3), n.start_line))
        head = f"\n### {path}"
        body = []
        for s in syms[:30]:
            sig = s.signature if s.kind in ("function", "method") else ""
            body.append(f"- {s.kind} `{s.qualified_name}{sig}` (L{s.start_line})")
        block = head + "\n" + "\n".join(body)
        # Stop adding files once we'd blow the budget.
        if sum(len(x) for x in out) + len(block) > max_chars:
            out.append(f"\n… ({len(files)} files total; outline truncated to fit)")
            break
        out.append(block)

    text = "\n".join(out)
    return text[:max_chars]
