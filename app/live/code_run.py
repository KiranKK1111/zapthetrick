"""
Live code verification: when the interviewer asks for a program, run the
generated solution in the sandbox before it's shown — in a language the
candidate lists on their resume — and repair it on failure.

The high-value guarantee is modest but real: the code the candidate sees
actually COMPILES and RUNS (no syntax errors, no crash, no garbled output).
Compiled languages (Java/Go) are compiled; interpreted ones (Python/JS/Ruby/PHP)
are executed. Anything the sandbox can't run (no toolchain, unsupported
language) is reported honestly as "not executed", never as verified.

Pure helpers here; the generate→verify→repair loop is driven by the orchestrator.
Deterministic + fail-open.
"""
from __future__ import annotations

import re

# Languages the throwaway sandbox can build + run (executor._LANGUAGES + the
# compiled Java/Go path). Order = preference when picking from a resume.
RUNNABLE: tuple[str, ...] = ("python", "java", "javascript", "go", "ruby", "php")

_ALIASES = {
    "py": "python", "python3": "python", "cpython": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "golang": "go", "c++": "cpp", "cs": "csharp", "c#": "csharp",
    "ts": "typescript",
}

# Phrases in the question that name a language (checked longest-first).
_Q_LANG: tuple[tuple[str, str], ...] = (
    ("javascript", "javascript"), ("typescript", "typescript"),
    ("python", "python"), ("java", "java"), ("golang", "go"),
    (" go ", "go"), ("ruby", "ruby"), (" php", "php"),
    ("c++", "cpp"), ("c#", "csharp"), ("kotlin", "kotlin"), ("rust", "rust"),
)

# Known programming languages we recognize in a resume skills list.
_KNOWN_LANGS = {
    "python", "java", "javascript", "typescript", "go", "ruby", "php",
    "cpp", "c", "csharp", "kotlin", "rust", "swift", "scala", "bash",
}


def normalize_lang(lang: str | None) -> str:
    low = (lang or "").strip().lower()
    return _ALIASES.get(low, low)


def language_in_question(question: str) -> str:
    """The language explicitly named in the question ('write a Java program'),
    or '' if none is named."""
    q = f" {(question or '').lower()} "
    for needle, canon in _Q_LANG:
        if needle in q:
            return canon
    return ""


def resume_languages(profile: dict | None) -> list[str]:
    """Programming languages the candidate lists (normalized, order preserved,
    deduped) from resume skills + project tech."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        c = normalize_lang(term)
        if c in _KNOWN_LANGS and c not in seen:
            seen.add(c)
            out.append(c)

    prof = profile if isinstance(profile, dict) else {}
    try:
        skills = prof.get("skills")
        if isinstance(skills, list):
            for s in skills:
                _add(str(s))
        elif isinstance(skills, str):
            for s in re.split(r"[,/|;]", skills):
                _add(s)
        for proj in (prof.get("projects") or []):
            if isinstance(proj, dict):
                tech = proj.get("tech")
                for t in (tech if isinstance(tech, list) else []):
                    _add(str(t))
    except Exception:  # noqa: BLE001
        return out
    return out


def pick_language(question: str, profile: dict | None,
                  default: str = "python") -> tuple[str, bool]:
    """Choose the target language for a coding answer and whether the sandbox
    can run it. Priority: language named in the question → a runnable resume
    language → any resume language → default. Returns (language, runnable)."""
    explicit = language_in_question(question)
    if explicit:
        return explicit, explicit in RUNNABLE
    resume = resume_languages(profile)
    for lang in resume:
        if lang in RUNNABLE:
            return lang, True
    if resume:
        return resume[0], resume[0] in RUNNABLE
    return default, default in RUNNABLE


_FENCE = re.compile(r"```([A-Za-z0-9_+#-]*)\s*\n(.*?)```", re.S)


def extract_code(answer: str, prefer_lang: str = "") -> tuple[str, str]:
    """The best fenced code block + its language from a markdown answer.
    Prefers a block whose tag matches `prefer_lang`; else the longest block.
    Returns ('', '') when there's no code block."""
    blocks = _FENCE.findall(answer or "")
    if not blocks:
        return "", ""
    pl = normalize_lang(prefer_lang)
    if pl:
        for tag, body in blocks:
            if normalize_lang(tag) == pl and body.strip():
                return body.strip(), pl
    # else the longest non-empty block
    tag, body = max(blocks, key=lambda tb: len(tb[1]))
    return body.strip(), (normalize_lang(tag) or pl)


def runnable_directive(language: str) -> str:
    """An instruction appended to the coding prompt so the generated code is
    actually EXECUTABLE — the sandbox needs an entry point to run it."""
    return (
        f"IMPORTANT: make the {language} code block fully self-contained and "
        f"runnable as-is: include the imports and a `main` (or top-level driver) "
        f"that exercises the solution on the example input(s) and prints the "
        f"result, so it compiles and runs without edits.")
