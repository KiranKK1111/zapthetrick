"""tree-sitter helpers — parser cache, a binding-agnostic Node wrapper, and
file→language detection.

The installed `tree-sitter-language-pack` binding exposes node accessors as
METHODS (``root_node()``, ``kind()``, ``start_position()`` → Point) rather than
properties. `_v` normalises both forms (call if callable), so this layer works
whether the binding uses methods or properties — the rest of the code stays
clean.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

log = logging.getLogger(__name__)

# tree-sitter parsers from this binding are "unsendable" — a parser created on
# one thread panics if used on another. We run parsing both on the event-loop
# thread (agent loop) and on worker threads (asyncio.to_thread in the context
# builder / AST edit tools), so the parser cache MUST be per-thread.
_TLS = threading.local()


def _v(x):
    """Resolve a zero-arg accessor that may be a method OR a property."""
    return x() if callable(x) else x


@lru_cache(maxsize=1)
def _available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        log.info("tree-sitter not available: %s", exc)
        return False


def _parser(language: str):
    """Thread-local cached parser for a language name, or None if unsupported/
    unavailable. Per-thread because the binding's parsers can't cross threads."""
    if not _available():
        return None
    cache = getattr(_TLS, "parsers", None)
    if cache is None:
        cache = _TLS.parsers = {}
    if language in cache:
        return cache[language]
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(language)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — unknown language
        parser = None
    cache[language] = parser
    return parser


# File extension → tree-sitter language name (language-pack naming).
EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".lua": "lua", ".dart": "dart", ".sh": "bash", ".bash": "bash",
}

# language-pack sometimes uses "c_sharp" instead of "csharp"; try both.
_LANG_ALIASES = {"csharp": ("csharp", "c_sharp")}


def language_for(path: str) -> str | None:
    i = path.rfind(".")
    if i < 0:
        return None
    return EXT_LANG.get(path[i:].lower())


def get_parser_for(language: str):
    """Parser for `language`, trying known aliases. None if unavailable."""
    for name in _LANG_ALIASES.get(language, (language,)):
        p = _parser(name)
        if p is not None:
            return p
    return None


class TSNode:
    """Thin, binding-agnostic wrapper over a tree-sitter node."""

    __slots__ = ("_n", "_src")

    def __init__(self, raw, src_bytes: bytes):
        self._n = raw
        self._src = src_bytes

    @property
    def kind(self) -> str:
        return _v(self._n.kind)

    @property
    def start_line(self) -> int:
        return _v(self._n.start_position).row + 1  # 1-based

    @property
    def end_line(self) -> int:
        return _v(self._n.end_position).row + 1

    @property
    def start_byte(self) -> int:
        return _v(self._n.start_byte)

    @property
    def end_byte(self) -> int:
        return _v(self._n.end_byte)

    @property
    def text(self) -> str:
        return self._src[self.start_byte:self.end_byte].decode("utf-8", "replace")

    def field(self, name: str) -> "TSNode | None":
        raw = self._n.child_by_field_name(name)
        return TSNode(raw, self._src) if raw is not None else None

    def children(self) -> list["TSNode"]:
        n = self._n
        count = _v(n.child_count)
        return [TSNode(n.child(i), self._src) for i in range(count)]

    def named_children(self) -> list["TSNode"]:
        n = self._n
        count = _v(n.named_child_count)
        return [TSNode(n.named_child(i), self._src) for i in range(count)]

    def descendants(self):
        """Pre-order walk over this node and all descendants (as TSNode)."""
        stack = [self]
        while stack:
            cur = stack.pop()
            yield cur
            kids = cur.children()
            # push reversed so we visit in source order
            stack.extend(reversed(kids))


def parse(path: str, source: str, language: str | None = None) -> tuple["TSNode | None", str | None]:
    """Parse `source` for the file `path`. Returns (root TSNode, language) or
    (None, language) when no parser is available for the language."""
    lang = language or language_for(path)
    if not lang:
        return None, None
    parser = get_parser_for(lang)
    if parser is None:
        return None, lang
    try:
        tree = parser.parse(source)   # this binding takes str
    except TypeError:
        # A binding variant that wants bytes.
        tree = parser.parse(source.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.info("parse failed for %s: %s", path, exc)
        return None, lang
    root = _v(tree.root_node)
    return TSNode(root, source.encode("utf-8")), lang
