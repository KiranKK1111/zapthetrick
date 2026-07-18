"""Framework resolvers — synthesize `route` nodes and `references` edges to the
handler functions/components that serve them (codegraph's framework layer).

Regex-based, run per file after AST extraction. Each detector emits route Nodes
into the FileExtract and records (route_id → handler_name) references, which the
builder binds to the handler's function node (same file preferred). Best-effort:
unmatched handlers simply yield a route node with no edge.
"""
from __future__ import annotations

import re

from .model import Node

# (decorator obj).(method)("path")  → FastAPI / Flask / Starlette / APIRouter
_PY_DECORATOR = re.compile(
    r"@\s*[\w.]+\.(get|post|put|patch|delete|options|head|route|websocket)\s*"
    r"\(\s*[rf]?['\"]([^'\"]+)['\"]"
    r"[\s\S]{0,400}?"                 # decorator args + any stacked decorators
    r"\n\s*(?:async\s+)?def\s+(\w+)",
    re.IGNORECASE,
)
# Django/Flask url maps:  path('route/', view)  |  re_path(r'...', view)  |  url(...)
_DJANGO_URL = re.compile(
    r"\b(?:path|re_path|url)\s*\(\s*[rf]?['\"]([^'\"]*)['\"]\s*,\s*([\w.]+)",
)
# Express / Koa / NestJS-ish:  app.get('/x', handler)  router.post('/x', handler)
_EXPRESS = re.compile(
    r"\b(?:app|router)\.(get|post|put|patch|delete|use|all)\s*\(\s*"
    r"[`'\"]([^`'\"]+)[`'\"]\s*,\s*([\w.]+)",
    re.IGNORECASE,
)
# React Router:  <Route path="/x" element={<Comp/>} />  or  component={Comp}
_REACT_ROUTE = re.compile(
    r"<Route\b[^>]*\bpath=['\"]([^'\"]+)['\"][^>]*?"
    r"(?:element=\{<\s*(\w+)|component=\{(\w+))",
)
# React Router object form:  { path: '/x', element: <Comp/> }  /  Component: Comp
_REACT_OBJ = re.compile(
    r"\bpath\s*:\s*['\"]([^'\"]+)['\"][\s\S]{0,80}?"
    r"(?:element\s*:\s*<\s*(\w+)|[Cc]omponent\s*:\s*(\w+))",
)


def _line_of(source: str, pos: int) -> int:
    return source.count("\n", 0, pos) + 1


def _last_segment(name: str) -> str:
    for sep in (".", "::"):
        if sep in name:
            name = name.split(sep)[-1]
    return name.strip()


def _add_route(fx, path: str, method: str, handler: str, line: int) -> None:
    method = (method or "ANY").upper()
    label = f"{method} {path}"
    rid = f"{fx.file.path}::route:{label}"
    fx.nodes.append(Node(
        id=rid, kind="route", name=path, qualified_name=label,
        path=fx.file.path, language=fx.language, start_line=line, end_line=line,
        signature=f"→ {handler}" if handler else "",
    ))
    fx.contains.append((fx.file.path, rid))
    if handler:
        fx.references.append((rid, _last_segment(handler), line))


def extract_frameworks(path: str, source: str, fx) -> None:
    """Append framework route nodes + references to `fx` (in place)."""
    lang = fx.language or ""
    try:
        if lang == "python":
            for m in _PY_DECORATOR.finditer(source):
                method, route, handler = m.group(1), m.group(2), m.group(3)
                _add_route(fx, route, "ANY" if method.lower() == "route" else method,
                           handler, _line_of(source, m.start()))
            for m in _DJANGO_URL.finditer(source):
                _add_route(fx, "/" + m.group(1).lstrip("/"), "ANY",
                           m.group(2), _line_of(source, m.start()))
        elif lang in ("javascript", "typescript", "tsx"):
            for m in _EXPRESS.finditer(source):
                _add_route(fx, m.group(2), m.group(1), m.group(3),
                           _line_of(source, m.start()))
            for m in _REACT_ROUTE.finditer(source):
                comp = m.group(2) or m.group(3)
                _add_route(fx, m.group(1), "ROUTE", comp, _line_of(source, m.start()))
            for m in _REACT_OBJ.finditer(source):
                comp = m.group(2) or m.group(3)
                _add_route(fx, m.group(1), "ROUTE", comp, _line_of(source, m.start()))
    except Exception:  # noqa: BLE001 — a regex hiccup must never break extraction
        pass
