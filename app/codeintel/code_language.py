"""Detect the target programming language + whether text is a coding problem.

Used when a user pastes a LeetCode / HackerRank / IDE screenshot and says
"solve this": the local vision model reads the image to text, and we then infer
the language they've SELECTED (the language chip, or the code stub's syntax) so
the assistant writes the solution in THAT language instead of stopping to ask.

Two signals, in priority order:
  1. an explicit language name in the text (the "Java" selector, "Python3", …);
  2. failing that, the syntax of any code stub present (`public int f(int[])`
     → Java, `class Solution:` → Python, `#include`/`vector<>` → C++, …).

Everything is best-effort and returns None when unsure — the caller then falls
back to the configured default language (never a blocking question).
"""
from __future__ import annotations

import re

# Explicit language mentions — the language selector / a spoken language name.
# Ordered so multi-word/again-specific tokens win (c++ before c, typescript
# before javascript-ish "script"). Values are canonical ids.
_EXPLICIT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpython\s*3?\b|\bpy3?\b", re.I), "python"),
    (re.compile(r"\btypescript\b|\bts\b", re.I), "typescript"),
    (re.compile(r"\bjavascript\b|\bnode(?:\.js)?\b", re.I), "javascript"),
    (re.compile(r"\bjava\b", re.I), "java"),
    # NOTE: `+`/`#` aren't word chars, so a trailing \b never matches after them
    # ("c++ " has no boundary between + and space). Anchor on the LEADING \b only.
    (re.compile(r"\bc\+\+|\bcpp\b", re.I), "cpp"),
    (re.compile(r"\bc#|\bc\s*sharp\b|\bcsharp\b", re.I), "csharp"),
    (re.compile(r"\bgo(?:lang)?\b", re.I), "go"),
    (re.compile(r"\brust\b", re.I), "rust"),
    (re.compile(r"\bkotlin\b", re.I), "kotlin"),
    (re.compile(r"\bswift\b", re.I), "swift"),
    (re.compile(r"\bruby\b", re.I), "ruby"),
    (re.compile(r"\bphp\b", re.I), "php"),
    (re.compile(r"\bscala\b", re.I), "scala"),
    (re.compile(r"\bdart\b", re.I), "dart"),
    # Extended set (SandboxLangPack.md Tier 1/2) — multi-char names only, to
    # avoid false positives from prose ("c"/"r" are inferred from the stub).
    (re.compile(r"\bpowershell\b|\bpwsh\b", re.I), "powershell"),
    (re.compile(r"\bbash\b|\bshell\b", re.I), "bash"),
    (re.compile(r"\bhaskell\b", re.I), "haskell"),
    (re.compile(r"\belixir\b", re.I), "elixir"),
    (re.compile(r"\berlang\b", re.I), "erlang"),
    (re.compile(r"\bjulia\b", re.I), "julia"),
    (re.compile(r"\bocaml\b", re.I), "ocaml"),
    (re.compile(r"\bf#|\bfsharp\b", re.I), "fsharp"),
    (re.compile(r"\bfortran\b", re.I), "fortran"),
    (re.compile(r"\bgroovy\b", re.I), "groovy"),
    (re.compile(r"\bperl\b", re.I), "perl"),
    (re.compile(r"\bjulia\b", re.I), "julia"),
    (re.compile(r"\blua\b", re.I), "lua"),
    (re.compile(r"\bnim\b", re.I), "nim"),
    (re.compile(r"\bzig\b", re.I), "zig"),
    (re.compile(r"\btcl\b", re.I), "tcl"),
    (re.compile(r"\boctave\b|\bmatlab\b", re.I), "octave"),
    (re.compile(r"\bsqlite3?\b|\bsql\b", re.I), "sql"),
    (re.compile(r"\br\s*script\b|\brlang\b", re.I), "r"),
    (re.compile(r"\bracket\b", re.I), "racket"),
    (re.compile(r"\bclojure\b", re.I), "clojure"),
    (re.compile(r"\bcommon\s*lisp\b|\blisp\b|\bsbcl\b", re.I), "lisp"),
    (re.compile(r"\bscheme\b|\bguile\b", re.I), "scheme"),
    (re.compile(r"\bprolog\b", re.I), "prolog"),
    (re.compile(r"\bpascal\b|\bdelphi\b", re.I), "pascal"),
    (re.compile(r"\bcobol\b", re.I), "cobol"),
    (re.compile(r"\bcrystal\b", re.I), "crystal"),
    (re.compile(r"\bobjective[-\s]?c\b|\bobjc\b", re.I), "objc"),
    (re.compile(r"\bvisual\s*basic\b|\bvb\.net\b|\bvbnet\b", re.I), "vbnet"),
    (re.compile(r"\bstandard\s*ml\b|\bsmlnj\b|\bsml\b", re.I), "sml"),
    (re.compile(r"\bsmalltalk\b", re.I), "smalltalk"),
    (re.compile(r"\bsolidity\b", re.I), "solidity"),
    (re.compile(r"\bchapel\b", re.I), "chapel"),
    (re.compile(r"\bgleam\b", re.I), "gleam"),
    (re.compile(r"\braku\b|\bperl\s*6\b", re.I), "raku"),
    (re.compile(r"\bgawk\b|\bawk\b", re.I), "awk"),
    (re.compile(r"\bzsh\b", re.I), "zsh"),
    (re.compile(r"\bodin\b", re.I), "odin"),
    (re.compile(r"\bmojo\b", re.I), "mojo"),
    (re.compile(r"\belm\b", re.I), "elm"),
]

# Code-stub syntax fingerprints, checked only when no name is stated. Each entry
# is (canonical_id, [regexes]); a single match is enough to call it.
_STUBS: list[tuple[str, list[re.Pattern]]] = [
    # Rust FIRST — its `impl`/`pub fn`/`Vec<>` are distinctive and would
    # otherwise be missed while `fn`/`->` overlap nothing here.
    ("rust", [re.compile(r"\bimpl\s+\w"), re.compile(r"\bpub\s+fn\b"),
              re.compile(r"\bfn\s+\w+\s*\("), re.compile(r"\bVec\s*<"),
              re.compile(r"\blet\s+mut\b"), re.compile(r"println!"),
              re.compile(r"->\s*i32\b"), re.compile(r"&self\b")]),
    # C# BEFORE Java (near-identical syntax): its Console./using System/`Main`
    # (capital) are the tell; Java uses System.out/`main` (lower).
    ("csharp", [re.compile(r"\bConsole\."), re.compile(r"\busing\s+System\b"),
                re.compile(r"\bstatic\s+void\s+Main\b"),
                re.compile(r"\bnamespace\s+\w")]),
    # Dart BEFORE Java — they look near-identical (C-family, typed, `class X`),
    # so without this a Dart stub (or a VLM that hallucinated Java from a Dart
    # screenshot) is misread as Java. Dart tells: lowercase-primitive generics
    # `List<int>` (illegal in Java, which needs `List<Integer>`), `void main()`
    # (Java is `main(String[])`), and `import 'dart:…'`.
    ("dart", [re.compile(r"import\s+'dart:"),
              re.compile(r"\bList<(?:int|double|bool|num)\b"),
              re.compile(r"\bvoid\s+main\s*\(\s*\)"),
              re.compile(r"\bprint\s*\([^)]*\)\s*;")]),
    ("java", [re.compile(r"\bpublic\s+(?:static\s+)?\w[\w<>\[\]]*\s+\w+\s*\("),
              re.compile(r"\bpublic\s+class\b"), re.compile(r"\bint\[\]\s"),
              re.compile(r"\bSystem\.out\b")]),
    # C++ BEFORE C, but WITHOUT bare `#include` (C uses it too) — require a
    # C++-only token; C then catches printf/scanf/<stdio.h>.
    ("cpp", [re.compile(r"\bstd::"), re.compile(r"\bvector\s*<"),
             re.compile(r"\bcout\b"), re.compile(r"\bcin\b"),
             re.compile(r"\bpublic:\b"), re.compile(r"\bendl\b"),
             re.compile(r"#include\s*<iostream"),
             re.compile(r"\busing\s+namespace\b")]),
    ("c", [re.compile(r"#include\s*<stdio"), re.compile(r"#include\s*<stdlib"),
           re.compile(r"\bprintf\s*\("), re.compile(r"\bscanf\s*\(")]),
    # Go's bare `func \w+(` also matched Swift's `func`, so Swift was misread as
    # Go. Use Go-DISTINCTIVE tells instead: `package main`, `fmt.`, the slice
    # type `[]int` (Swift is `[Int]`), and `:=`. A LeetCode Go stub always has a
    # slice param/return, so this still fires; Swift's `[Int]`/`->` never do.
    ("go", [re.compile(r"\bpackage\s+main\b"), re.compile(r"\bfmt\."),
            re.compile(r"\[\]\w"), re.compile(r":="),
            re.compile(r"\bfunc\s+\w+\s*\([^)]*\)\s*\[\]")]),
    ("typescript", [re.compile(r":\s*\w+\[\]\s*\)"),
                    re.compile(r"\bfunction\s+\w+\s*\([^)]*:\s*\w")]),
    ("javascript", [re.compile(r"\bfunction\s+\w+\s*\("),
                    re.compile(r"\bconsole\.log\b"),
                    re.compile(r"=>"), re.compile(r"\b(?:const|let|var)\s+\w+\s*=")]),
    ("kotlin", [re.compile(r"\bfun\s+\w+\s*\("), re.compile(r"\bval\s+\w"),
                re.compile(r"\bprintln\s*\(")]),
    ("swift", [re.compile(r"\bfunc\s+\w+\s*\([^)]*:\s*\["),
               re.compile(r"\blet\s+\w+\s*=\s*"), re.compile(r"\bprint\s*\(")]),
    # Lisp-family (parenthesised prefix syntax) BEFORE Python: a small local VLM
    # transcribing e.g. a Racket screenshot often HALLUCINATES a `def trap(...):`
    # alongside the real `(define …)` stub. `(define`/`(defn`/`(ns` never appear
    # in real Python, so checking these first makes the REAL syntax win the tie
    # instead of the hallucinated Python. (The "<Lang> Auto" chip + explicit name
    # already run before any stub; this covers the case OCR missed the chip.)
    ("clojure", [re.compile(r"\(defn\s"), re.compile(r"\(ns\s"),
                 re.compile(r"#\(")]),
    # Racket: `define/contract`, contract combinators `(-> (listof …) …)`, its
    # `exact-integer?`-style predicates, and `#lang` — none of which the plain
    # `\(define\s` pattern caught (it missed `(define/contract`).
    ("racket", [re.compile(r"#lang\b"),
                re.compile(r"\(define(?:/contract|-|\s)"),
                re.compile(r"\(require\b"), re.compile(r"\(->\s*\("),
                re.compile(r"\(listof\b"),
                re.compile(r"\bexact-(?:integer|nonnegative-integer|rational)\?")]),
    # Scala BEFORE Python: its `def` would otherwise match the Python stub. Its
    # `object X`, typed-`def … : T =`, and `Array[…]` (square, vs Kotlin's `<…>`)
    # are distinctive.
    ("scala", [re.compile(r"\bobject\s+\w+"),
               re.compile(r"\bdef\s+\w+[^=\n]*:\s*\w+\s*="),
               re.compile(r"\bArray\[")]),
    # Python's `def` must carry its trailing colon, else Ruby's `def x … end`
    # (no colon) would be misread as Python.
    ("python", [re.compile(r"\bclass\s+\w+\s*:"),
                re.compile(r"\bdef\s+\w+\s*\([^)]*\)\s*(?:->[^\n:]+)?:"),
                re.compile(r"\bself\b"), re.compile(r"\bprint\s*\(")]),
    # Elixir + Erlang BEFORE Ruby: Ruby's `def … end` pattern (re.S) also matches
    # Elixir's `def … do … end` and Erlang clauses, so checking Ruby first
    # misread every Elixir/Erlang stub as Ruby. Their `do`/`defmodule`/`-module`
    # are distinctive and never appear in real Ruby.
    ("elixir", [re.compile(r"\bdefmodule\b"),
                re.compile(r"\bdef\s+\w+\s*\([^)\n]*\)\s+do\b"),
                re.compile(r"\|>"), re.compile(r"\bIO\.puts\b")]),
    ("erlang", [re.compile(r"-spec\s"), re.compile(r"-module\s*\("),
                re.compile(r"-export\s*\("),
                re.compile(r"\bunicode:unicode_binary\b"),
                re.compile(r"->\s*\n.*\bend\b", re.S)]),
    ("ruby", [re.compile(r"\bdef\s+\w+.*\bend\b", re.S), re.compile(r"\bputs\b")]),
    ("haskell", [re.compile(r"::\s*\[?\w+\]?\s*->"), re.compile(r"\bwhere\b\s*$", re.M),
                 re.compile(r"^\w+\s+::\s", re.M)]),
    ("fsharp", [re.compile(r"\blet\s+rec\b"), re.compile(r"\blet\s+\w+\s*=.*\bfun\b"),
                re.compile(r"\bprintfn\b")]),
    ("ocaml", [re.compile(r"\blet\s+rec\b.*="), re.compile(r"\bmatch\b.*\bwith\b"),
               re.compile(r";;\s*$", re.M)]),
]

# Markers that this text is a competitive-programming / coding problem — used to
# decide whether to fall back to a DEFAULT language (vs. leaving it to the
# normal clarifier). Kept broad but code-specific.
_PROBLEM_RE = re.compile(
    r"\bleetcode\b|\bhackerrank\b|\bclass\s+Solution\b|\bTest\s*case\b|"
    r"\bTest\s*Result\b|\bConstraints?\s*:|\bExample\s*\d*\s*:|"
    r"\bInput\s*:.*\bOutput\s*:|\bnums\b|\breturn\s+the\b",
    re.I | re.S,
)


# LeetCode/HackerRank render the selected language as "<Language> Auto" at the
# top of the code editor. That chip is the single most reliable signal — and it
# covers short/symbol names (C, R, D, V, C++, C#, F#) that are ambiguous in
# prose. Capture 1–2 words (incl. + # . -) immediately before "Auto".
# The optional 2nd word must NOT swallow "Auto" itself (the `(?!Auto\b)`), else
# "Erlang Auto" is captured as a 2-word chip and the required trailing "Auto"
# then has nothing to match.
_CHIP_AUTO_RE = re.compile(
    r"([A-Za-z][\w#+.\-]*(?:\s+(?!Auto\b)[A-Za-z][\w#+.\-]*)?)\s+Auto\b")


def _chip_language(text: str) -> tuple[str | None, str | None]:
    """(canonical id, exact chip label) from a '<Lang> Auto' editor chip."""
    from app.sandbox.lang_registry import canonical  # noqa: PLC0415
    for m in _CHIP_AUTO_RE.finditer(text or ""):
        chip = m.group(1).strip()
        cid = canonical(chip)
        if cid is None and " " in chip:
            cid = canonical(chip.split()[-1])   # e.g. "Common Lisp" → "Lisp"
        if cid is not None:
            return cid, chip
    return None, None


def detect_language(text: str) -> str | None:
    """Best-effort canonical language id: the editor's language chip first, then
    an explicit name, then code-stub syntax. None when nothing is recognisable."""
    if not text:
        return None
    # 0) the "<Lang> Auto" chip — highest precision, handles C/R/D/V/C++/C#.
    cid, _chip = _chip_language(text)
    if cid is not None:
        return cid
    # 1) explicit name (the selected language / a stated one).
    for pat, lang in _EXPLICIT:
        if pat.search(text):
            return lang
    # 2) infer from the code stub's syntax.
    for lang, pats in _STUBS:
        if any(p.search(text) for p in pats):
            return lang
    return None


# A language name sitting next to a REQUEST cue ("in Swift", "the Dart code").
# Deliberately narrow so prose like "go through this" / "as a class" isn't read
# as Go — the language token must follow a cue word or precede a code noun.
_REQUEST_CUE_RE = re.compile(
    r"\b(?:in|using|with|into|to|write(?:\s+it)?\s+in|give\s+me\s+(?:the\s+)?)"
    r"\s+([A-Za-z][\w#+]{0,14})\b"
    r"|\b([A-Za-z][\w#+]{0,14})\s+"
    r"(?:solution|code|program|version|language|lang|please)\b",
    re.I)


def requested_language(text: str) -> str | None:
    """The language the USER explicitly asked for in prose ("solve this in
    Swift", "give me the Dart code"). This is a direct instruction and OUTRANKS
    any chip/OCR/stub read — so a typed request always wins over a mis-read
    screenshot. Only a language name adjacent to a request cue counts (so "go
    through this" is not misread as Go). None when the user named none."""
    if not text:
        return None
    from app.sandbox.lang_registry import canonical  # noqa: PLC0415

    def _resolve(tok: str) -> str | None:
        if not tok or tok.lower() in _REQUEST_STOPWORDS:
            return None
        cid = canonical(tok)
        if cid is None:
            # Resolve version-y / symbolic names (python3, c#, golang) via the
            # explicit-name table, matched against JUST this token.
            for pat, lang in _EXPLICIT:
                if pat.fullmatch(tok):
                    return lang
        return cid

    for m in _REQUEST_CUE_RE.finditer(text):
        cid = _resolve((m.group(1) or m.group(2) or "").strip())
        if cid is not None:
            return cid
    # Bare-name fallback for a TERSE request (e.g. the composer just says
    # "swift" or "use dart" alongside a screenshot). Only for short inputs — a
    # bare token in long prose ("a run in Go country") must NOT be read as a
    # request, so the cue-based pass above stays authoritative there.
    words = re.findall(r"[A-Za-z][\w#+]*", text)
    if 1 <= len(words) <= 3:
        for w in words:
            cid = _resolve(w)
            if cid is not None:
                return cid
    return None


# Common words that follow a cue but aren't languages (avoid "in the …" noise).
_REQUEST_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "it", "one", "any", "some", "which",
    "your", "my", "our", "code", "solution", "program", "language", "same",
    "another", "different", "each", "both", "form", "order", "place",
})


# Canonical id -> the DISPLAY label used in a solve directive. Version-specific
# labels (Python3 vs Python2) are resolved separately below — LeetCode/HackerRank
# treat them as distinct languages with different syntax.
_CANON_TO_LABEL = {
    "python": "Python3", "java": "Java", "cpp": "C++", "csharp": "C#",
    "javascript": "JavaScript", "typescript": "TypeScript", "go": "Go",
    "rust": "Rust", "kotlin": "Kotlin", "swift": "Swift", "ruby": "Ruby",
    "php": "PHP", "scala": "Scala", "dart": "Dart", "c": "C",
    "powershell": "PowerShell", "bash": "Bash", "haskell": "Haskell",
    "elixir": "Elixir", "erlang": "Erlang", "julia": "Julia", "ocaml": "OCaml",
    "fsharp": "F#", "fortran": "Fortran", "groovy": "Groovy", "perl": "Perl",
    "lua": "Lua", "nim": "Nim", "zig": "Zig", "tcl": "Tcl", "octave": "Octave",
    "sql": "SQL", "r": "R", "racket": "Racket", "clojure": "Clojure",
    "lisp": "Common Lisp", "scheme": "Scheme", "prolog": "Prolog",
    "pascal": "Pascal", "cobol": "COBOL", "crystal": "Crystal", "d": "D",
    "vlang": "V", "objc": "Objective-C", "vbnet": "Visual Basic",
    "sml": "Standard ML", "smalltalk": "Smalltalk", "solidity": "Solidity",
    "chapel": "Chapel", "gleam": "Gleam", "raku": "Raku", "awk": "AWK",
    "zsh": "Zsh", "odin": "Odin", "mojo": "Mojo", "elm": "Elm", "hack": "Hack",
    "red": "Red", "hy": "Hy", "make": "Makefile",
}

# Version/variant chips whose exact form matters (different syntax per version).
_VERSION_LABELS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpython\s*3\b|\bpy3\b", re.I), "Python3"),
    (re.compile(r"\bpython\s*2\b|\bpy2\b", re.I), "Python2"),
    (re.compile(r"\bc\+\+\s*(?:20|23|17|14|11)\b", re.I), ""),   # keep as shown
    (re.compile(r"\bc99\b|\bc11\b|\bc17\b|\bgnu\s*c\b", re.I), ""),
]


def detect_language_label(text: str) -> str | None:
    """The language label to write the solution IN, preserving version when the
    chip shows one (e.g. 'Python3' vs 'Python2', 'C++17'). Falls back to the
    canonical language's default label. None when nothing is recognisable."""
    if not text:
        return None
    for pat, lab in _VERSION_LABELS:
        m = pat.search(text)
        if m:
            return lab or m.group(0).strip()
    # The editor chip carries the exact label (e.g. "C++", "Objective-C").
    cid, chip = _chip_language(text)
    if cid is not None:
        return _CANON_TO_LABEL.get(cid, chip)
    canon = detect_language(text)
    return _CANON_TO_LABEL.get(canon) if canon else None


def looks_like_coding_problem(text: str) -> bool:
    """True when the text reads like a coding/algorithm problem to solve — the
    signal that a missing language should default rather than be asked."""
    return bool(text) and bool(_PROBLEM_RE.search(text))
