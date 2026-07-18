"""Build a CodeGraph from a project's source files.

    files: Iterable[(path, source)]  ->  CodeGraph

Phases: (1) filter to source files, skipping vendored dirs/oversized blobs;
(2) per-file extraction; (3) reference resolution — bind calls to functions,
class bases to classes, and internal imports to files — by name/path.
"""
from __future__ import annotations

from collections.abc import Iterable

from .extract import extract_file
from .model import CodeGraph, Edge
from .tsutil import language_for

# Directories whose contents are never part of "the project's" code.
_SKIP_DIRS = {
    "node_modules", ".git", ".hg", ".svn", "dist", "build", "out", "bin",
    "obj", "target", "vendor", "venv", ".venv", "env", "__pycache__",
    ".idea", ".vscode", "bower_components", ".next", ".nuxt", ".svelte-kit",
    "coverage", ".pytest_cache", ".mypy_cache", ".tox", "site-packages",
    "Pods", ".gradle", ".dart_tool", "Debug", "Release",
}
_MAX_FILE_BYTES = 1_500_000      # skip generated/minified monsters
_MAX_FILES = 4_000               # cap so a giant repo can't run forever


def _is_source(path: str) -> bool:
    p = path.replace("\\", "/")
    if any(seg in _SKIP_DIRS for seg in p.split("/")):
        return False
    name = p.rsplit("/", 1)[-1]
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return False
    return language_for(p) is not None


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def build_code_graph(files: Iterable[tuple[str, str]]) -> CodeGraph:
    g = CodeGraph()

    # --- Phase 1+2: filter + extract ---
    extracts = []
    seen = 0
    for raw_path, source in files:
        path = _norm(raw_path)
        if not _is_source(path):
            continue
        if seen >= _MAX_FILES:
            g.errors.append(f"file cap reached ({_MAX_FILES}); remaining skipped")
            break
        if source is None or len(source.encode("utf-8", "ignore")) > _MAX_FILE_BYTES:
            g.files_skipped += 1
            continue
        seen += 1
        fx = extract_file(path, source)
        extracts.append(fx)
        g.files_parsed += 1
        g.languages[fx.language or "?"] = g.languages.get(fx.language or "?", 0) + 1
        if fx.error:
            g.errors.append(f"{path}: {fx.error}")

        g.add_node(fx.file)
        for n in fx.nodes:
            g.add_node(n)
        for parent_id, child_id in fx.contains:
            g.add_edge(Edge(src=parent_id, dst=child_id, kind="contains"))

    # --- Phase 3: reference resolution ---
    # Index callable + type symbols by short name for name-based binding.
    callable_by_name: dict[str, list] = {}
    type_by_name: dict[str, list] = {}
    for n in g.nodes.values():
        if n.kind in ("function", "method"):
            callable_by_name.setdefault(n.name, []).append(n)
        elif n.kind in ("class", "interface", "struct", "trait", "enum"):
            type_by_name.setdefault(n.name, []).append(n)

    file_paths = [n.path for n in g.files]

    def _pick(cands: list, same_file: str) -> list:
        """Resolve an ambiguous name to edge target(s): prefer a same-file
        match; else a unique project match; else up to 3 (bounded fan-out)."""
        if not cands:
            return []
        local = [c for c in cands if c.path == same_file]
        if len(local) == 1:
            return local
        if len(cands) == 1:
            return cands
        return cands[:3]

    for fx in extracts:
        # calls → functions/methods, import-aware: if the callee was imported
        # `from <module> import <callee>`, prefer the symbol in that module's
        # file before falling back to name-only matching (kills false edges to
        # same-named symbols elsewhere).
        for caller_id, callee, line in fx.calls:
            cands = callable_by_name.get(callee, [])
            mod = fx.imported.get(callee)
            if mod and len(cands) > 1:
                tgt_path = _resolve_import(mod, fx.file.path, file_paths)
                if tgt_path:
                    pref = [c for c in cands if c.path == tgt_path]
                    if pref:
                        cands = pref
            for tgt in _pick(cands, fx.file.path):
                g.add_edge(Edge(src=caller_id, dst=tgt.id, kind="calls", line=line))
        # class bases → classes/interfaces
        for class_id, base in fx.extends:
            for tgt in _pick(type_by_name.get(base, []), fx.file.path):
                g.add_edge(Edge(src=class_id, dst=tgt.id, kind="extends"))
        # framework routes → their handler function/component
        for src_id, target, line in fx.references:
            cands = callable_by_name.get(target, []) or type_by_name.get(target, [])
            for tgt in _pick(cands, fx.file.path):
                g.add_edge(Edge(src=src_id, dst=tgt.id, kind="references", line=line))

    # internal imports → file→file edges (best-effort path matching)
    for fx in extracts:
        for mod, line in fx.imports:
            tgt = _resolve_import(mod, fx.file.path, file_paths)
            if tgt and tgt != fx.file.path:
                g.add_edge(Edge(src=fx.file.path, dst=tgt, kind="imports", line=line))

    return g


def _resolve_import(module: str, importer: str, file_paths: list[str]) -> str | None:
    """Best-effort: map an import statement to a project file path by suffix
    match. Handles Python dotted modules + relative imports; for other
    languages, matches a quoted/relative path fragment if present."""
    m = (module or "").strip()
    if not m:
        return None

    # Python-style: leading dots = relative; dots = path separators.
    if m and (m[0] == "." or all(c.isidentifier() or c == "." for c in m.replace(".", "a"))):
        dots = len(m) - len(m.lstrip("."))
        rest = m.lstrip(".").replace(".", "/")
        if dots:  # relative to importer's package
            base = importer.rsplit("/", 1)[0]
            for _ in range(max(0, dots - 1)):
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            cand_stem = f"{base}/{rest}".strip("/") if rest else base
        else:
            cand_stem = rest
        for suffix in (f"{cand_stem}.py", f"{cand_stem}/__init__.py"):
            hit = _suffix_match(suffix, file_paths)
            if hit:
                return hit

    # Generic: an import that contains a relative path fragment ("./x", "../y/z").
    frag = ""
    for tok in m.replace("'", '"').split('"'):
        if "/" in tok or tok.startswith("."):
            frag = tok.strip()
            break
    if frag:
        frag = frag.lstrip("./")
        for ext in (".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".py", ""):
            hit = _suffix_match(frag + ext, file_paths)
            if hit:
                return hit
    return None


def _suffix_match(suffix: str, file_paths: list[str]) -> str | None:
    suffix = suffix.strip("/")
    if not suffix:
        return None
    for p in file_paths:
        if p == suffix or p.endswith("/" + suffix):
            return p
    return None
