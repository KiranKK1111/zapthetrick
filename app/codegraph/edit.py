"""AST-aware source edits (Phase 6, report #21).

Three structural edits that are SAFER than blind text replacement because they
work over the tree-sitter parse tree:

  • `rename_symbol`  — rename an identifier everywhere it appears AS AN
    IDENTIFIER (so matches inside strings / comments are left alone — the bug
    with a naive regex rename).
  • `insert_import`  — add an import/use/include line in the right place (after
    the existing imports, or after the file's prelude), and never duplicate one.
  • `add_method`     — insert a method/member into a named class/struct body,
    indented to match, before the body's close.

Each works on a SOURCE STRING and returns an `EditResult` (no I/O), so it's pure
and unit-testable; the agent tools in `app/agent/tools.py` wrap them with
read→apply→write. On anything it can't do safely (no parser for the language,
symbol/class not found) it returns `ok=False` with a reason, so the caller can
fall back to the plain `edit` tool.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tsutil import parse

# Identifier-ish node kinds across languages (NOT string/comment/number nodes —
# that's the whole point: we only touch real identifiers).
_IDENT_KINDS = {
    "identifier", "type_identifier", "field_identifier", "property_identifier",
    "shorthand_property_identifier", "shorthand_property_identifier_pattern",
    "name", "constant", "global_variable", "instance_variable",
    "simple_identifier", "label_name", "namespace_identifier",
}

# Import/use/include statement kinds.
_IMPORT_KINDS = {
    "import_statement", "import_from_statement", "import_declaration",
    "import_spec", "import_spec_list", "using_directive", "use_declaration",
    "preproc_include", "package_import", "require_call",
}

# Class/struct/interface kinds whose body can hold methods/members.
_CLASS_KINDS = {
    "class_definition", "class_declaration", "struct_item",
    "struct_specifier", "class_specifier", "interface_declaration",
    "object_declaration", "impl_item", "trait_item", "enum_declaration",
}

# Comment/docstring-ish kinds we skip when locating a file's "prelude".
_PRELUDE_SKIP = {"comment", "line_comment", "block_comment", "shebang",
                 "expression_statement", "package_clause", "package_declaration"}


@dataclass
class EditResult:
    ok: bool
    source: str            # modified source (or the original on failure)
    changed: bool
    detail: str

    @classmethod
    def fail(cls, source: str, detail: str) -> "EditResult":
        return cls(False, source, False, detail)


def _name_of(node) -> str | None:
    """Best-effort symbol name for a class/def node (its `name` field or first
    identifier child)."""
    f = node.field("name")
    if f is not None:
        return f.text
    for c in node.children():
        if c.kind in _IDENT_KINDS:
            return c.text
    return None


# --------------------------------------------------------------------------
# rename
# --------------------------------------------------------------------------
def rename_symbol(source: str, old: str, new: str, *,
                  path: str = "file.py", language: str | None = None) -> EditResult:
    """Rename every IDENTIFIER occurrence of `old` to `new` (file-scoped).
    Skips matches inside strings/comments (they aren't identifier nodes)."""
    if not old or not new:
        return EditResult.fail(source, "old and new names are required")
    if old == new:
        return EditResult.fail(source, "old and new names are identical")
    root, lang = parse(path, source, language)
    if root is None:
        return EditResult.fail(source, f"no tree-sitter parser for {lang or path}")
    spans: list[tuple[int, int]] = []
    for node in root.descendants():
        if node.kind in _IDENT_KINDS and node.text == old:
            spans.append((node.start_byte, node.end_byte))
    if not spans:
        return EditResult.fail(source, f"identifier '{old}' not found")
    b = source.encode("utf-8")
    repl = new.encode("utf-8")
    for s, e in sorted(spans, reverse=True):     # back-to-front keeps offsets
        b = b[:s] + repl + b[e:]
    return EditResult(True, b.decode("utf-8", "replace"), True,
                      f"renamed {len(spans)} occurrence(s) of '{old}' to '{new}'")


# --------------------------------------------------------------------------
# insert import
# --------------------------------------------------------------------------
def _prelude_end(root, b: bytes) -> int:
    """Byte offset after a file's leading comments / shebang / package line /
    a Python module docstring — where a first import should go."""
    pos = 0
    for child in root.named_children():
        if child.kind in _PRELUDE_SKIP:
            # A module-level string expression = docstring → keep skipping.
            pos = child.end_byte
            continue
        break
    return pos


def insert_import(source: str, import_line: str, *,
                  path: str = "file.py", language: str | None = None) -> EditResult:
    """Insert `import_line` after the existing imports (or after the prelude),
    de-duplicated. `import_line` is the full statement, e.g.
    'import os' / 'from x import y' / 'use crate::z;' / '#include <v>'."""
    line = (import_line or "").strip()
    if not line:
        return EditResult.fail(source, "import_line is required")
    if line in (l.strip() for l in source.splitlines()):
        return EditResult.fail(source, "import already present")
    root, lang = parse(path, source, language)
    if root is None:
        return EditResult.fail(source, f"no tree-sitter parser for {lang or path}")

    last_import_end: int | None = None
    for child in root.named_children():
        if child.kind in _IMPORT_KINDS:
            last_import_end = child.end_byte
    b = source.encode("utf-8")
    if last_import_end is not None:
        ins = last_import_end
        chunk = ("\n" + line).encode("utf-8")
    else:
        ins = _prelude_end(root, b)
        prefix = "" if ins == 0 else "\n"
        chunk = (prefix + line + "\n").encode("utf-8")
    nb = b[:ins] + chunk + b[ins:]
    return EditResult(True, nb.decode("utf-8", "replace"), True,
                      f"inserted import: {line}")


# --------------------------------------------------------------------------
# add method / member
# --------------------------------------------------------------------------
def _find_class(root, class_name: str):
    for node in root.descendants():
        if node.kind in _CLASS_KINDS and _name_of(node) == class_name:
            return node
    return None


def _class_body(node):
    for fname in ("body", "declaration_list", "field_declaration_list"):
        f = node.field(fname)
        if f is not None:
            return f
    # Fallback: the last block-ish child.
    for c in reversed(node.children()):
        if c.kind in ("block", "class_body", "declaration_list",
                      "field_declaration_list", "enum_body"):
            return c
    return None


def _line_indent(source: str, byte_off: int) -> str:
    """Whitespace at the start of the line containing `byte_off`."""
    text = source.encode("utf-8")[:byte_off].decode("utf-8", "replace")
    line_start = text.rfind("\n") + 1
    line = source[line_start:]
    return line[: len(line) - len(line.lstrip(" \t"))]


def add_method(source: str, class_name: str, method_code: str, *,
               path: str = "file.py", language: str | None = None) -> EditResult:
    """Insert `method_code` into `class_name`'s body, indented to match, just
    before the body closes. Works for Python (indented block) and brace
    languages (inserts before the closing `}`)."""
    if not class_name or not (method_code or "").strip():
        return EditResult.fail(source, "class_name and method_code are required")
    root, lang = parse(path, source, language)
    if root is None:
        return EditResult.fail(source, f"no tree-sitter parser for {lang or path}")
    cls = _find_class(root, class_name)
    if cls is None:
        return EditResult.fail(source, f"class/struct '{class_name}' not found")
    body = _class_body(cls)
    if body is None:
        return EditResult.fail(source, f"could not locate body of '{class_name}'")

    members = body.named_children()
    method = method_code.strip("\n")
    b = source.encode("utf-8")

    # Indentation: match the first member, else one level past the class header.
    if members:
        indent = _line_indent(source, members[0].start_byte)
    else:
        indent = _line_indent(source, cls.start_byte) + "    "
    indented = "\n".join((indent + ln if ln.strip() else ln)
                         for ln in method.splitlines())

    brace_lang = any(c.kind == "}" for c in body.children())
    if brace_lang:
        # Insert before the closing brace.
        close = None
        for c in body.children():
            if c.kind == "}":
                close = c
        ins = close.start_byte if close is not None else body.end_byte
        chunk = (f"\n{indented}\n").encode("utf-8")
        nb = b[:ins] + chunk + b[ins:]
    else:
        # Python-style block: insert after the last member (or after the
        # block's start for an empty body).
        ins = members[-1].end_byte if members else body.end_byte
        chunk = (f"\n\n{indented}").encode("utf-8")
        nb = b[:ins] + chunk + b[ins:]

    return EditResult(True, nb.decode("utf-8", "replace"), True,
                      f"added member to '{class_name}'")


__all__ = ["EditResult", "rename_symbol", "insert_import", "add_method"]
