"""Per-file symbol + relationship extraction.

Python is parsed with the stdlib `ast` (precise, always available, gives exact
calls/inheritance). Every other language is parsed with tree-sitter via a
generic, language-agnostic AST walk that keys off tree-sitter's fairly uniform
node-kind names (``*_definition`` / ``*_declaration`` / ``*_item`` …). Both
produce the same `FileExtract` so the builder treats them uniformly.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .model import Node
from .tsutil import language_for, parse


@dataclass
class FileExtract:
    file: Node
    nodes: list[Node] = field(default_factory=list)
    contains: list[tuple[str, str]] = field(default_factory=list)        # (parent_id, child_id)
    calls: list[tuple[str, str, int]] = field(default_factory=list)      # (caller_id, callee_name, line)
    imports: list[tuple[str, int]] = field(default_factory=list)         # (module_str, line)
    extends: list[tuple[str, str]] = field(default_factory=list)         # (class_id, base_name)
    references: list[tuple[str, str, int]] = field(default_factory=list)  # (src_id, target_name, line)
    imported: dict[str, str] = field(default_factory=dict)               # local name → source module
    language: str = ""
    error: str = ""


def _file_node(path: str, language: str) -> Node:
    return Node(id=path, kind="file", name=path.rsplit("/", 1)[-1],
                qualified_name=path, path=path, language=language)


# ----------------------------------------------------------------------------
# Python — stdlib ast (precise, dependency-free)
# ----------------------------------------------------------------------------
def extract_python_ast(path: str, source: str) -> FileExtract:
    fx = FileExtract(file=_file_node(path, "python"), language="python")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        fx.error = f"syntax error: {exc}"
        return fx

    def nid(qual: str) -> str:
        return f"{path}::{qual}"

    def base_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return base_name(node.func)
        return None

    def sig_of(fn: ast.AST) -> str:
        try:
            args = fn.args  # type: ignore[attr-defined]
            parts = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
            if args.vararg:
                parts.append("*" + args.vararg.arg)
            parts += [a.arg for a in args.kwonlyargs]
            if args.kwarg:
                parts.append("**" + args.kwarg.arg)
            return "(" + ", ".join(parts) + ")"
        except Exception:  # noqa: BLE001
            return ""

    def visit(node: ast.AST, parent_id: str, qual_prefix: str, enclosing_fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{qual_prefix}{child.name}"
                cid = nid(qual)
                kind = "method" if "." in qual else "function"
                fx.nodes.append(Node(
                    id=cid, kind=kind, name=child.name, qualified_name=qual,
                    path=path, language="python", start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno) or child.lineno,
                    signature=sig_of(child),
                ))
                fx.contains.append((parent_id, cid))
                visit(child, cid, f"{qual}.", cid)
            elif isinstance(child, ast.ClassDef):
                qual = f"{qual_prefix}{child.name}"
                cid = nid(qual)
                fx.nodes.append(Node(
                    id=cid, kind="class", name=child.name, qualified_name=qual,
                    path=path, language="python", start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno) or child.lineno,
                ))
                fx.contains.append((parent_id, cid))
                for b in child.bases:
                    bn = base_name(b)
                    if bn:
                        fx.extends.append((cid, bn))
                visit(child, cid, f"{qual}.", enclosing_fn)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    fx.imports.append((alias.name, child.lineno))
                    fx.imported[alias.asname or alias.name.split(".")[0]] = alias.name
                visit(child, parent_id, qual_prefix, enclosing_fn)
            elif isinstance(child, ast.ImportFrom):
                mod = ("." * (child.level or 0)) + (child.module or "")
                fx.imports.append((mod, child.lineno))
                for alias in child.names:
                    fx.imported[alias.asname or alias.name] = mod
                visit(child, parent_id, qual_prefix, enclosing_fn)
            elif isinstance(child, ast.Call):
                if enclosing_fn:
                    bn = base_name(child.func)
                    if bn:
                        fx.calls.append((enclosing_fn, bn, getattr(child, "lineno", 0)))
                visit(child, parent_id, qual_prefix, enclosing_fn)
            else:
                visit(child, parent_id, qual_prefix, enclosing_fn)

    visit(tree, path, "", None)
    return fx


# ----------------------------------------------------------------------------
# Everything else — generic tree-sitter walk
# ----------------------------------------------------------------------------
# tree-sitter node-kind → our node kind. Broad, cross-language.
_DEF_KINDS: dict[str, str] = {
    # functions / methods
    "function_definition": "function", "function_declaration": "function",
    "function_item": "function", "method_definition": "method",
    "method_declaration": "method", "constructor_declaration": "method",
    "function_signature": "function", "func_literal": "function",
    "arrow_function": "function", "generator_function_declaration": "function",
    "subroutine": "function", "fn_item": "function",
    # types / containers
    "class_definition": "class", "class_declaration": "class",
    "class_specifier": "class", "struct_item": "struct",
    "struct_specifier": "struct", "struct_declaration": "struct",
    "interface_declaration": "interface", "trait_item": "trait",
    "enum_declaration": "enum", "enum_item": "enum", "enum_specifier": "enum",
    "type_declaration": "class", "object_declaration": "class",
    "module": "module", "impl_item": "class",
}
_CONTAINER_KINDS = {"class", "interface", "struct", "trait", "enum", "module"}
_IMPORT_KINDS = {
    "import_statement", "import_from_statement", "import_declaration",
    "use_declaration", "import_spec", "preproc_include", "using_directive",
    "package_import_declaration", "import", "import_clause",
}
_CALL_KINDS = {
    "call", "call_expression", "method_invocation", "function_call_expression",
    "invocation_expression", "call_expr",
}
_IDENT_KINDS = {"identifier", "type_identifier", "field_identifier",
                "constant", "scoped_identifier", "dotted_name", "name"}


def _node_name(node) -> str:
    """Best-effort short name for a definition node."""
    f = node.field("name")
    if f is not None:
        return f.text.split("\n")[0][:120]
    # else: first identifier-ish named child
    for c in node.named_children():
        if c.kind in _IDENT_KINDS:
            return c.text.split("\n")[0][:120]
    return ""


def _callee_name(call_node) -> str:
    """Short callee name from a call node (last identifier segment)."""
    fn = call_node.field("function") or call_node.field("name")
    target = fn if fn is not None else call_node
    last = ""
    for d in target.descendants():
        if d.kind in _IDENT_KINDS:
            last = d.text
    if not last:
        last = target.text
    # keep the final segment of a.b.c / a::b / a->b
    for sep in ("::", ".", "->"):
        if sep in last:
            last = last.split(sep)[-1]
    return last.strip().split("(")[0][:120]


def extract_treesitter(path: str, source: str) -> FileExtract:
    lang = language_for(path) or ""
    fx = FileExtract(file=_file_node(path, lang), language=lang)
    root, lang2 = parse(path, source, lang or None)
    fx.language = lang2 or lang
    fx.file.language = fx.language
    if root is None:
        return fx  # no parser → file node only

    def nid(qual: str) -> str:
        return f"{path}::{qual}"

    def walk(node, parent_id: str, qual_prefix: str, enclosing_fn: str | None) -> None:
        for child in node.children():
            k = child.kind
            our = _DEF_KINDS.get(k)
            if our:
                name = _node_name(child)
                if not name:
                    walk(child, parent_id, qual_prefix, enclosing_fn)
                    continue
                qual = f"{qual_prefix}{name}"
                cid = nid(qual)
                params = child.field("parameters")
                fx.nodes.append(Node(
                    id=cid, kind=our, name=name, qualified_name=qual, path=path,
                    language=fx.language, start_line=child.start_line,
                    end_line=child.end_line,
                    signature=(params.text[:200] if params is not None else ""),
                ))
                fx.contains.append((parent_id, cid))
                new_fn = cid if our in ("function", "method") else enclosing_fn
                new_prefix = f"{qual}." if our in _CONTAINER_KINDS else qual_prefix
                # methods nest under containers too (so calls attribute correctly)
                if our in ("function", "method"):
                    new_prefix = f"{qual}."
                walk(child, cid, new_prefix, new_fn)
            elif k in _IMPORT_KINDS:
                fx.imports.append((child.text.split("\n")[0][:200], child.start_line))
            elif k in _CALL_KINDS:
                if enclosing_fn:
                    cn = _callee_name(child)
                    if cn:
                        fx.calls.append((enclosing_fn, cn, child.start_line))
                walk(child, parent_id, qual_prefix, enclosing_fn)
            else:
                walk(child, parent_id, qual_prefix, enclosing_fn)

    walk(root, path, "", None)
    return fx


def extract_file(path: str, source: str) -> FileExtract:
    """Dispatch: Python → ast; other languages → tree-sitter. Then layer on
    framework route resolution (FastAPI/Django/Flask/Express/React Router)."""
    if (language_for(path) or "") == "python":
        fx = extract_python_ast(path, source)
    else:
        fx = extract_treesitter(path, source)
    from .frameworks import extract_frameworks
    extract_frameworks(path, source, fx)
    return fx
