"""Static-analysis helpers for the architecture fitness-function suite.

These helpers read the `app/` tree as TEXT and parse it with `ast` — they never
IMPORT app modules, so the whole suite runs with no network, no models, no keys,
and no heavy native deps (torch / onnxruntime / parselmouth). That keeps it fast
and CI-safe, exactly like the offline eval suite.

Roadmap: implements the "Architecture Invariants" (L8623) / "Architectural
Fitness Functions" (L11522) enforcement plan in ImplementationRoadmap.md
(section "🛡️ Guardrail Enforcement Plan"). Phase 0.
"""
from __future__ import annotations

import ast
import functools
import pathlib

# tests/architecture/_scan.py -> parents[2] == zapthetrick_be
BE_ROOT = pathlib.Path(__file__).resolve().parents[2]
APP_ROOT = BE_ROOT / "app"

_IGNORE_DIRS = {"__pycache__"}


def top_level_packages() -> set[str]:
    """Directory names directly under app/ that are Python packages."""
    out: set[str] = set()
    for child in APP_ROOT.iterdir():
        if child.is_dir() and child.name not in _IGNORE_DIRS:
            if (child / "__init__.py").exists():
                out.add(child.name)
    return out


def package_modules(pkg: str) -> set[str]:
    """Module stems (no .py, no __init__) directly inside app/<pkg>/."""
    pdir = APP_ROOT / pkg
    return {
        p.stem
        for p in pdir.glob("*.py")
        if p.stem != "__init__"
    }


def app_python_files() -> list[pathlib.Path]:
    """Every .py under app/, excluding __pycache__."""
    return [
        p
        for p in APP_ROOT.rglob("*.py")
        if not any(part in _IGNORE_DIRS for part in p.parts)
    ]


def module_dotted(path: pathlib.Path) -> str:
    """app/agents/planner.py -> 'app.agents.planner'."""
    rel = path.relative_to(BE_ROOT).with_suffix("")
    return ".".join(rel.parts)


def package_of(path: pathlib.Path) -> str:
    """First directory under app/ for a file, or '(root)' if directly in app/."""
    rel = path.relative_to(APP_ROOT)
    return rel.parts[0] if len(rel.parts) >= 2 else "(root)"


@functools.lru_cache(maxsize=None)
def _parse(path_str: str) -> ast.Module:
    return ast.parse(pathlib.Path(path_str).read_text(encoding="utf-8"), filename=path_str)


def parse(path: pathlib.Path) -> ast.Module:
    return _parse(str(path))


def _resolve_from(current_module: str, node: ast.ImportFrom) -> str | None:
    """Resolve a `from ... import` target to an absolute dotted module name."""
    if node.level == 0:
        return node.module
    parts = current_module.split(".")
    anchor = parts[: len(parts) - node.level]
    if node.module:
        anchor = anchor + node.module.split(".")
    return ".".join(anchor) if anchor else None


def imported_app_modules(path: pathlib.Path) -> set[str]:
    """Absolute dotted `app.*` modules imported by this file (abs + relative)."""
    current = module_dotted(path)
    out: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from(current, node)
            if resolved and resolved.startswith("app."):
                out.add(resolved)
    return out


def cross_package_edges() -> set[tuple[str, str]]:
    """Set of (src_pkg -> dst_pkg) import edges between top-level app packages.

    Same-package imports are ignored. This is the dependency graph the
    import-boundary guardrail freezes as a baseline.
    """
    edges: set[tuple[str, str]] = set()
    for path in app_python_files():
        src = package_of(path)
        for mod in imported_app_modules(path):
            parts = mod.split(".")
            dst = parts[1] if len(parts) >= 2 else "(root)"
            if dst != src:
                edges.add((src, dst))
    return edges


def dataclass_fields(path: pathlib.Path, class_name: str) -> set[str]:
    """Names of annotated (dataclass-style) fields on a class, via AST."""
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
            return fields
    return set()


def has_class(path: pathlib.Path, class_name: str) -> bool:
    return any(
        isinstance(n, ast.ClassDef) and n.name == class_name
        for n in ast.walk(parse(path))
    )


def module_level_functions(path: pathlib.Path) -> set[str]:
    return {
        n.name
        for n in parse(path).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
