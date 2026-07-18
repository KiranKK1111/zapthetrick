"""Canonical project stack detection (ArchitectureVerdict Phase 2).

One entry point answering "what is this project?" — language, framework,
build tool, package manager, database hints — from manifests + file-extension
evidence. Previously this knowledge was scattered (codegraph framework
regexes, per-agent manifest checks); the design doc's ask (Clarification
Elimination / Workspace Awareness): an uploaded `pom.xml` means NEVER asking
"which language?".

Two inputs:
  * `detect_stack_from_members(members)` — (name, text) pairs, e.g. from
    `codegraph.archive.iter_source_members` on an uploaded zip;
  * `detect_stack(root)` — a materialized workspace directory.

Output [StackProfile] carries per-fact confidence + evidence strings, ready to
fill the clarifier's RequirementMatrix with source="attachment".

Deterministic, dependency-free, fail-open (never raises; empty profile means
"couldn't tell" and changes nothing downstream).
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

# Manifest name → (language, package_manager, build_tool). Framework comes
# from the manifest CONTENT below.
_MANIFESTS: dict[str, tuple[str, str, str]] = {
    "package.json": ("javascript", "npm", "npm"),
    "requirements.txt": ("python", "pip", "pip"),
    "pyproject.toml": ("python", "pip", "pip"),
    "pipfile": ("python", "pipenv", "pipenv"),
    "pom.xml": ("java", "maven", "maven"),
    "build.gradle": ("java", "gradle", "gradle"),
    "build.gradle.kts": ("kotlin", "gradle", "gradle"),
    "go.mod": ("go", "go modules", "go"),
    "cargo.toml": ("rust", "cargo", "cargo"),
    "pubspec.yaml": ("dart", "pub", "flutter"),
    "composer.json": ("php", "composer", "composer"),
    "gemfile": ("ruby", "bundler", "bundler"),
    "mix.exs": ("elixir", "hex", "mix"),
}
_CSPROJ_RE = re.compile(r"\.(cs|fs)proj$", re.IGNORECASE)

# Framework cues searched inside manifest text (dependency names win over
# prose). Ordered most-specific first per language family.
_FRAMEWORK_CUES: tuple[tuple[str, str], ...] = (
    ("spring-boot", "spring boot"),
    ("springframework", "spring"),
    ("quarkus", "quarkus"),
    ("micronaut", "micronaut"),
    ("fastapi", "fastapi"),
    ("django", "django"),
    ("flask", "flask"),
    ("next", "next.js"),
    ("nuxt", "nuxt"),
    ("@angular/core", "angular"),
    ("react-native", "react native"),
    ("react", "react"),
    ("vue", "vue"),
    ("svelte", "svelte"),
    ("express", "express"),
    ("nestjs", "nestjs"),
    ("@nestjs/core", "nestjs"),
    ("koa", "koa"),
    ("laravel", "laravel"),
    ("rails", "rails"),
    ("actix", "actix"),
    ("axum", "axum"),
    ("gin-gonic", "gin"),
    ("fiber", "fiber"),
    ("flutter", "flutter"),
    ("phoenix", "phoenix"),
    ("asp.net", "asp.net"),
    ("microsoft.aspnetcore", "asp.net"),
)
_DB_CUES: tuple[tuple[str, str], ...] = (
    ("postgres", "postgresql"), ("psycopg", "postgresql"), ("pgvector", "postgresql"),
    ("mysql", "mysql"), ("mariadb", "mariadb"),
    ("mongodb", "mongodb"), ("mongoose", "mongodb"), ("pymongo", "mongodb"),
    ("sqlite", "sqlite"), ("redis", "redis"),
    ("sqlalchemy", "sql"), ("prisma", "sql"), ("hibernate", "sql"),
)
# Extension histogram fallback (only when no manifest names a language).
_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
    ".kt": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "c#", ".dart": "dart", ".swift": "swift",
    ".cpp": "c++", ".cc": "c++", ".c": "c", ".ex": "elixir",
}
_MAX_MEMBERS = 4000          # bound directory walks / member scans


@dataclass
class StackProfile:
    """Canonical detected stack with per-fact confidence + evidence trail."""
    language: str | None = None
    framework: str | None = None
    build_tool: str | None = None
    package_manager: str | None = None
    db_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0              # of the LANGUAGE fact
    evidence: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.language or self.framework)

    def slots(self) -> dict:
        """The clarifier-slot view (only filled facts), for RequirementMatrix
        filling with source='attachment' and for suppression."""
        out: dict = {}
        if self.language:
            out["language"] = self.language
        if self.framework:
            out["framework"] = self.framework
        return out

    def as_dict(self) -> dict:
        return {
            "language": self.language, "framework": self.framework,
            "build_tool": self.build_tool,
            "package_manager": self.package_manager,
            "db_hints": list(self.db_hints),
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
        }


def _scan_manifest(name: str, text: str, p: StackProfile) -> None:
    """Fold one manifest file's evidence into the profile."""
    base = name.rsplit("/", 1)[-1].lower()
    hit = _MANIFESTS.get(base)
    if hit is None and _CSPROJ_RE.search(base):
        hit = ("c#", "nuget", "dotnet")
    if hit:
        lang, pkg, build = hit
        # TypeScript refinement: package.json + a tsconfig seen elsewhere is
        # handled by the caller (needs cross-file knowledge); here dependency
        # text saying "typescript" is enough.
        if base == "package.json" and "typescript" in (text or "").lower():
            lang = "typescript"
        if not p.language:
            p.language, p.confidence = lang, 0.9
            p.evidence.append(f"manifest:{base}")
        p.package_manager = p.package_manager or pkg
        p.build_tool = p.build_tool or build
    low = (text or "").lower()
    if not p.framework:
        for cue, fw in _FRAMEWORK_CUES:
            if cue in low:
                p.framework = fw
                p.evidence.append(f"framework_cue:{cue} in {base}")
                break
    for cue, db in _DB_CUES:
        if cue in low and db not in p.db_hints:
            p.db_hints.append(db)


def detect_stack_from_members(members) -> StackProfile:
    """Detect from (name, text) pairs — archive members, editor buffers, etc.
    Manifests are authoritative; the extension histogram fills language only
    when no manifest named one."""
    p = StackProfile()
    try:
        exts: Counter = Counter()
        saw_tsconfig = False
        for i, (name, text) in enumerate(members):
            if i >= _MAX_MEMBERS:
                break
            base = (name or "").rsplit("/", 1)[-1].lower()
            if base == "tsconfig.json":
                saw_tsconfig = True
            if base in _MANIFESTS or _CSPROJ_RE.search(base):
                _scan_manifest(name or "", text or "", p)
            ext = pathlib.PurePosixPath(name or "").suffix.lower()
            if ext in _EXT_LANG:
                exts[ext] += 1
        if saw_tsconfig and p.language == "javascript":
            p.language = "typescript"
            p.evidence.append("manifest:tsconfig.json")
        if not p.language and exts:
            ext, n = exts.most_common(1)[0]
            p.language = _EXT_LANG[ext]
            p.confidence = 0.7 if n >= 3 else 0.5
            p.evidence.append(f"extension_histogram:{ext}x{n}")
    except Exception:  # noqa: BLE001 — an empty profile changes nothing
        pass
    return p


def detect_stack(root: str | pathlib.Path) -> StackProfile:
    """Detect from a materialized workspace directory (bounded walk)."""
    def _iter():
        try:
            base = pathlib.Path(root)
            count = 0
            for f in base.rglob("*"):
                if count >= _MAX_MEMBERS:
                    return
                if not f.is_file():
                    continue
                count += 1
                name = f.relative_to(base).as_posix()
                low = f.name.lower()
                text = ""
                if low in _MANIFESTS or _CSPROJ_RE.search(low) \
                        or low == "tsconfig.json":
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")[:200_000]
                    except Exception:  # noqa: BLE001
                        text = ""
                yield name, text
        except Exception:  # noqa: BLE001
            return
    return detect_stack_from_members(_iter())


def detect_stack_from_archive(data: bytes, filename: str) -> StackProfile:
    """Detect from raw uploaded archive bytes (zip/tar/7z) by reusing the
    codegraph member iterator, widened with a manifest-only pass: the source
    iterator skips non-code files, so manifests are read directly here."""
    p = StackProfile()
    try:
        import io
        import zipfile
        names_text: list[tuple[str, str]] = []
        if (filename or "").lower().endswith(".zip") or (
                data[:2] == b"PK"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist()[:_MAX_MEMBERS]:
                    base = info.filename.rsplit("/", 1)[-1].lower()
                    if info.is_dir():
                        continue
                    text = ""
                    if base in _MANIFESTS or _CSPROJ_RE.search(base) \
                            or base == "tsconfig.json":
                        try:
                            text = zf.read(info)[:200_000].decode(
                                "utf-8", errors="ignore")
                        except Exception:  # noqa: BLE001
                            text = ""
                    names_text.append((info.filename, text))
            return detect_stack_from_members(names_text)
        # Non-zip archives: fall back to the codegraph source iterator (code
        # files only — manifests may be missed, extensions still classify).
        from app.codegraph.archive import iter_source_members
        return detect_stack_from_members(iter_source_members(data, filename))
    except Exception:  # noqa: BLE001
        return p


__all__ = ["StackProfile", "detect_stack", "detect_stack_from_members",
           "detect_stack_from_archive"]
