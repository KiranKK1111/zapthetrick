"""Language polyglot layer — Architecture.md §"Polyglot".

Three pieces:

    idioms.py     — per-language style hints for the LLM prompt
    formatters.py — best-effort code formatting (black, gofmt, prettier, …)
    linters.py    — fast lint pass (ruff, eslint, …)

The formatters / linters call out to the local toolchain when
available; when not, they're no-ops. Architecture.md commits to
18+ languages — the scaffold supports the most common 14 and the
rest fall through to no-op idioms.
"""
from .idioms import idiom_hint, SUPPORTED_LANGUAGES
from .formatters import format_code
from .linters import lint_code


__all__ = ["idiom_hint", "format_code", "lint_code", "SUPPORTED_LANGUAGES"]
