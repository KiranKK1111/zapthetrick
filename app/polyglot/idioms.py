"""Idiom hints for the LLM prompt.

Each language gets a one-line style nudge appended to the code-
generation prompt. The hints are short on purpose — the model
knows the language; we're just steering it away from common
non-idiomatic patterns (raw indexing in Python, `var` in modern
JS, etc.).
"""
from __future__ import annotations


SUPPORTED_LANGUAGES = (
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "cpp",
    "csharp",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "scala",
    "dart",
    "elixir",
    "haskell",
    "ocaml",
)


_IDIOM_HINTS: dict[str, str] = {
    "python": "Use list comprehensions and f-strings. Prefer pathlib over os.path. Type-hint everything.",
    "javascript": "Use `const` / `let`, never `var`. Async/await over `.then`. Optional chaining for nullables.",
    "typescript": "Strict types — no `any`. Use `unknown` when truly unknown. Prefer `type` aliases over `interface` for simple shapes.",
    "java": "Use records for value types. Streams + Optional where they read well. Avoid raw collections.",
    "go": "Idiomatic Go — short receiver names, lower-case package, return errors as the last value. No exceptions.",
    "rust": "Use `?` for error propagation. Prefer iterators over indexed loops. Borrow check before writing — don't `.clone()` to silence the compiler.",
    "c": "Initialize every variable. Bounds-check every array access. Free what you malloc.",
    "cpp": "RAII for ownership. `std::unique_ptr` over raw new. Use ranges/views where they read well.",
    "csharp": "Use `var` for obvious types. `async`/`await` end-to-end. Records for immutable data.",
    "kotlin": "Data classes for value types. Null-safety with `?` and `!!`. Sealed classes for ADT-style enums.",
    "swift": "Value types by default. `let` over `var`. Guard-let for early returns.",
    "ruby": "Block syntax, idiomatic enumerable methods. `attr_*` instead of manual getters.",
    "php": "PSR-12 style. Typed properties. Use `??` for nullables.",
    "scala": "Pure functions, immutability, comprehensions. Avoid Java-style nulls.",
    "dart": "`final` over `var` where possible. Null-safety with `?`. Idiomatic widget composition.",
}


def idiom_hint(language: str) -> str:
    return _IDIOM_HINTS.get((language or "").lower(), "")


__all__ = ["idiom_hint", "SUPPORTED_LANGUAGES"]
