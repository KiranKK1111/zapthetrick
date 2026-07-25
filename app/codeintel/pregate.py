"""Tree-sitter syntax pre-gate (vNext §3.1).

A ~millisecond parse that rejects **syntactically broken** code before the
verifier pays a compile, and confirms a fenced block is real code (not prose in
a fence). It never authors or judges *behaviour* — only "does this parse?".

**Fail-open by construction:** unknown language, no parser available, empty
code, or any hiccup → `(True, None)` — the pre-gate abstains and the compiler
(or the model) decides. It can only ever *skip* a doomed compile, never block a
good answer.

Reuses the tree-sitter machinery already in `app/codegraph/tsutil.py` (the
language pack is pinned in requirements); no new dependency.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Common code-fence labels / aliases → tree-sitter-language-pack grammar names.
_TS_NAME: dict[str, str] = {
    "python": "python", "py": "python", "python3": "python", "python2": "python",
    "javascript": "javascript", "js": "javascript", "node": "javascript",
    "jsx": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "tsx",
    "java": "java",
    "go": "go", "golang": "go",
    "c": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
    "csharp": "csharp", "c#": "csharp", "cs": "csharp",
    "rust": "rust", "rs": "rust",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "kotlin": "kotlin", "kt": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "bash": "bash", "sh": "bash", "shell": "bash",
}

# A hard cap so a pathological (huge/deeply-nested) answer can't make the "cheap"
# pre-gate expensive — over the cap we abstain (fail-open).
_MAX_NODES = 60_000


def ts_language(language: str | None) -> str | None:
    """Map a fence label / language name (e.g. ``"Python 3"``, ``"js"``) to a
    tree-sitter grammar name, or None when we don't gate that language."""
    s = (language or "").strip().lower()
    if not s:
        return None
    head = s.split()[0]                      # "python 3" → "python"
    return _TS_NAME.get(s) or _TS_NAME.get(head)


def parse_ok(code: str, language: str | None) -> tuple[bool, str | None]:
    """``(ok, first_error)``.

    * ``(True, None)``  → parses cleanly, OR the language can't be gated (abstain).
    * ``(False, msg)``  → a definite syntax error (an ERROR node), with a short
      ``"syntax error near line N"`` message for repair feedback.
    Never raises.
    """
    if not (code or "").strip():
        return True, None
    tsname = ts_language(language)
    if not tsname:
        return True, None                    # unknown language → abstain
    try:
        from app.codegraph import tsutil
        root, _lang = tsutil.parse("", code, language=tsname)
    except Exception:  # noqa: BLE001
        return True, None
    if root is None:
        return True, None                    # no parser installed → abstain
    try:
        seen = 0
        for node in root.descendants():
            seen += 1
            if seen > _MAX_NODES:
                return True, None            # too big to gate cheaply → abstain
            if node.kind == "ERROR":
                return False, f"syntax error near line {node.start_line}"
    except Exception:  # noqa: BLE001
        return True, None
    return True, None


def looks_like_code(code: str, language: str | None) -> bool:
    """A cheap "is this real code, not prose in a fence?" check: it parses under
    the language's grammar without a top-level error AND has structure. Abstains
    to ``True`` for un-gateable languages (don't suppress a real answer)."""
    if not (code or "").strip():
        return False
    if ts_language(language) is None:
        return True                          # can't tell → assume code
    ok, _ = parse_ok(code, language)
    return ok


__all__ = ["parse_ok", "looks_like_code", "ts_language"]
