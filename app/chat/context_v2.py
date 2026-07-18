"""Context engineering v2 (P2-3, report_2 §P2-3).

Phase-5's `context_builder` ranks files and packs head-excerpts. This module
layers the next tier of context engineering on top so weak free models stay
oriented in a large repo:

  • build_repo_digest()  — a HIERARCHICAL file→dir→project digest (languages,
        entry points, per-directory key symbols), cached under
        `.zapthetrick/digest.md` keyed by a tree fingerprint so it's only
        rebuilt when files actually change. A cheap "map of the repo" preamble.
  • compress_source()    — squeeze a file down to its SIGNATURES plus the
        relevant HUNKS (windows around lines that mention the task's terms),
        instead of a blind head excerpt — far more signal per token.
  • rank_with_recency()  — boost recently-edited files (codegraph + name overlap
        from Phase 5, now + recent-edit signal).
  • build_context_v2()   — the assembled preamble: recency-aware ranking packed
        with compression, optionally prefixed by the cached repo digest.

All deterministic + offline (no LLM, no network). Reuses Phase-5 primitives
(`rank_files`, `_outline`, `_read_workspace`, `ContextBudget/Result`).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

from app.chat.context_builder import (
    _ENTRYPOINTS,
    _SKIP_DIRS,
    _est_tokens,
    _is_source,
    _outline,
    _stem,
    ContextBudget,
    ContextResult,
    RankedFile,
    rank_files,
)

_MAX_FILE_BYTES = 1_500_000
_MAX_WALK_FILES = 4_000
_DIGEST_DIR = ".zapthetrick"
_DIGEST_FILE = "digest.md"
_DIGEST_FP = "digest.fp"

# extension → language label (for the project language breakdown)
_LANG = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C/C++", ".cpp": "C++", ".cc": "C++", ".cs": "C#",
    ".kt": "Kotlin", ".swift": "Swift", ".dart": "Dart", ".scala": "Scala",
    ".sql": "SQL", ".sh": "Shell", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".md": "Markdown", ".html": "HTML", ".css": "CSS",
}

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


# --------------------------------------------------------------------------
# Read the workspace with file metadata (content + mtime + size).
# --------------------------------------------------------------------------
@dataclass
class _FileMeta:
    rel: str
    content: str
    mtime: float
    size: int


def _read_workspace_meta(workspace: str) -> list[_FileMeta]:
    root = os.path.realpath(workspace)
    out: list[_FileMeta] = []
    if not os.path.isdir(root):
        return out
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS]
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if not _is_source(rel):
                continue
            try:
                st = os.stat(full)
                if st.st_size > _MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    out.append(_FileMeta(rel, f.read(), st.st_mtime, st.st_size))
            except OSError:
                continue
            if len(out) >= _MAX_WALK_FILES:
                return out
    return out


# --------------------------------------------------------------------------
# Hierarchical repo digest (file → dir → project), cached.
# --------------------------------------------------------------------------
@dataclass
class RepoDigest:
    text: str = ""
    fingerprint: str = ""
    file_count: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    entrypoints: list[str] = field(default_factory=list)
    cached: bool = False


def _fingerprint(metas: list[_FileMeta]) -> str:
    h = hashlib.sha1()
    for m in sorted(metas, key=lambda x: x.rel):
        h.update(f"{m.rel}:{m.size}:{int(m.mtime)}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _topdir(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else "."


def _render_digest(metas: list[_FileMeta], *, max_chars: int) -> tuple[
        str, dict[str, int], list[str]]:
    """Build the project→dir overview text + language breakdown + entrypoints."""
    langs: dict[str, int] = {}
    entrypoints: list[str] = []
    by_dir: dict[str, list[_FileMeta]] = {}
    for m in metas:
        ext = os.path.splitext(m.rel)[1].lower()
        lang = _LANG.get(ext)
        if lang:
            langs[lang] = langs.get(lang, 0) + 1
        if _stem(m.rel) in _ENTRYPOINTS:
            entrypoints.append(m.rel)
        by_dir.setdefault(os.path.dirname(m.rel) or ".", []).append(m)

    lang_str = ", ".join(f"{k} ({v})" for k, v in
                         sorted(langs.items(), key=lambda kv: -kv[1])[:6])
    lines = [
        "REPOSITORY MAP (hierarchical digest — the project's shape so you can "
        "navigate without exploring blindly):",
        f"- Files: {len(metas)}   Languages: {lang_str or 'n/a'}",
    ]
    if entrypoints:
        lines.append("- Entry points: " + ", ".join(sorted(entrypoints)[:8]))

    # Per-directory: file count + a few key symbols, biggest dirs first.
    lines.append("- Directories:")
    dirs_sorted = sorted(by_dir.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for d, files in dirs_sorted[:40]:
        # collect a handful of representative symbols across the dir's files
        syms: list[str] = []
        for fm in files[:6]:
            outline = _outline(fm.content, max_syms=4)
            for ln in outline.splitlines():
                name = _first_symbol_name(ln)
                if name and name not in syms:
                    syms.append(name)
                if len(syms) >= 6:
                    break
            if len(syms) >= 6:
                break
        label = d if d != "." else "(root)"
        sym_str = f" — {', '.join(syms)}" if syms else ""
        lines.append(f"    {label}/  [{len(files)} files]{sym_str}")
        if sum(len(x) for x in lines) > max_chars:
            lines.append("    … (truncated)")
            break

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… (truncated)"
    return text, langs, entrypoints


def _first_symbol_name(decl_line: str) -> str:
    """Pull the declared name out of a signature line (best-effort)."""
    m = re.search(
        r"\b(?:def|class|func|function|interface|struct|trait|enum|type|fn|"
        r"impl|module|namespace)\s+([A-Za-z_][A-Za-z0-9_]*)", decl_line)
    return m.group(1) if m else ""


def _digest_paths(workspace: str) -> tuple[str, str]:
    d = os.path.join(os.path.realpath(workspace), _DIGEST_DIR)
    return os.path.join(d, _DIGEST_FILE), os.path.join(d, _DIGEST_FP)


def build_repo_digest(workspace: str, *, max_chars: int = 4000,
                      use_cache: bool = True) -> RepoDigest:
    """Build (or reuse a cached) hierarchical repo digest for `workspace`.

    Cached under `.zapthetrick/digest.md` with a fingerprint sidecar; rebuilt
    only when the file set / sizes / mtimes change. Returns an empty digest for
    an empty workspace."""
    metas = _read_workspace_meta(workspace)
    if not metas:
        return RepoDigest()
    fp = _fingerprint(metas)
    dpath, fppath = _digest_paths(workspace)

    if use_cache:
        try:
            with open(fppath, encoding="utf-8") as f:
                if f.read().strip() == fp:
                    with open(dpath, encoding="utf-8") as df:
                        text = df.read()
                    return RepoDigest(text=text, fingerprint=fp,
                                      file_count=len(metas), cached=True)
        except OSError:
            pass

    text, langs, entrypoints = _render_digest(metas, max_chars=max_chars)
    if use_cache:
        try:
            os.makedirs(os.path.dirname(dpath), exist_ok=True)
            with open(dpath, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            with open(fppath, "w", encoding="utf-8", newline="\n") as f:
                f.write(fp)
        except OSError:
            pass
    return RepoDigest(text=text, fingerprint=fp, file_count=len(metas),
                      languages=langs, entrypoints=entrypoints, cached=False)


# --------------------------------------------------------------------------
# Read compression: signatures + relevant hunks.
# --------------------------------------------------------------------------
def compress_source(source: str, task: str = "", *, max_lines: int = 80,
                    hunk_window: int = 4, max_hunks: int = 6) -> str:
    """Squeeze `source` to its signatures + hunks relevant to `task`.

    Returns a compact view: an OUTLINE (def/class/… signatures) followed by a
    few WINDOWS around the lines that mention the task's identifiers. Falls back
    to a head excerpt when there's nothing to anchor on. Far more signal per
    token than a blind head slice for a big file."""
    src = source or ""
    lines = src.splitlines()
    if len(lines) <= max_lines:
        return src  # small enough — return as-is

    outline = _outline(src, max_syms=40)
    terms = {t.lower() for t in _TOKEN.findall(task or "")} - _STOPWORDS
    hunks: list[str] = []
    if terms:
        hit_lines = [i for i, ln in enumerate(lines)
                     if any(t in ln.lower() for t in terms)]
        # merge nearby hits into windows
        windows: list[tuple[int, int]] = []
        for i in hit_lines:
            lo, hi = max(0, i - hunk_window), min(len(lines), i + hunk_window + 1)
            if windows and lo <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
            else:
                windows.append((lo, hi))
            if len(windows) >= max_hunks:
                break
        for lo, hi in windows[:max_hunks]:
            body = "\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi))
            hunks.append(f"  [lines {lo + 1}-{hi}]\n{body}")

    parts: list[str] = []
    if outline:
        parts.append("SIGNATURES:\n" + outline)
    if hunks:
        parts.append("RELEVANT EXCERPTS:\n" + "\n".join(hunks))
    if not parts:
        head = "\n".join(lines[:max_lines])
        return head + "\n…"
    return "\n\n".join(parts)


_STOPWORDS = {
    "the", "and", "for", "this", "that", "with", "from", "into", "fix", "add",
    "use", "make", "code", "file", "files", "function", "method", "class",
    "please", "should", "would", "could", "have", "has", "are", "was", "but",
}


# --------------------------------------------------------------------------
# Recency-aware ranking.
# --------------------------------------------------------------------------
def rank_with_recency(ranked: list[RankedFile], mtimes: dict[str, float],
                      *, weight: float = 2.0) -> list[RankedFile]:
    """Boost recently-modified files. Newest gets the full `weight`, scaled
    linearly down to the oldest. Stable: re-sorts by the boosted score."""
    if not ranked or not mtimes:
        return ranked
    present = [mtimes[r.path] for r in ranked if r.path in mtimes]
    if not present:
        return ranked
    lo, hi = min(present), max(present)
    span = (hi - lo) or 1.0
    out: list[RankedFile] = []
    for r in ranked:
        mt = mtimes.get(r.path)
        if mt is None:
            out.append(r)
            continue
        boost = weight * ((mt - lo) / span)
        reason = r.reason
        if boost >= weight * 0.6:
            reason = (reason + "; recently edited").strip("; ")
        out.append(RankedFile(r.path, r.score + boost, reason))
    out.sort(key=lambda r: (-r.score, r.path))
    return out


# --------------------------------------------------------------------------
# Assembled v2 context: recency ranking + compression (+ optional digest).
# --------------------------------------------------------------------------
def build_context_v2(workspace: str, task: str, *,
                     budget: ContextBudget | None = None,
                     include_digest: bool = True) -> ContextResult:
    """Recency-aware, compression-packed project context, optionally prefixed by
    the hierarchical repo digest. Drop-in replacement for `build_context`."""
    budget = budget or ContextBudget()
    metas = _read_workspace_meta(workspace)
    if not metas:
        return ContextResult()
    files = [(m.rel, m.content) for m in metas]
    mtimes = {m.rel: m.mtime for m in metas}

    ranked = rank_files(files, task, limit=budget.max_files)
    ranked = rank_with_recency(ranked, mtimes)
    if not ranked:
        # still give the model the map of the repo if we have one
        if include_digest:
            dig = build_repo_digest(workspace)
            if dig.text:
                return ContextResult(text=dig.text + "\n",
                                     tokens=_est_tokens(dig.text))
        return ContextResult()

    by_path = dict(files)
    parts: list[str] = []
    used = 0

    if include_digest:
        dig = build_repo_digest(workspace)
        if dig.text:
            block = dig.text + "\n"
            parts.append(block)
            used += _est_tokens(block)

    header = (
        "\nRELEVANT FILES (ranked for THIS task — signatures + the excerpts "
        "that matter; read a file in full before editing it):\n"
    )
    parts.append(header)
    used += _est_tokens(header)
    included: list[str] = []
    truncated = False

    for i, rf in enumerate(ranked):
        source = by_path.get(rf.path, "")
        if i < budget.outline_only_after:
            body = compress_source(source, task, max_lines=budget.head_lines)
        else:
            body = _outline(source)
            if not body:
                continue
        block = f"\n### {rf.path}  — {rf.reason}\n```\n{body}\n```\n"
        cost = _est_tokens(block)
        if used + cost > budget.max_tokens:
            truncated = True
            outline = _outline(source)
            if outline:
                alt = f"\n### {rf.path}  — {rf.reason} (outline)\n```\n{outline}\n```\n"
                if used + _est_tokens(alt) <= budget.max_tokens:
                    parts.append(alt)
                    used += _est_tokens(alt)
                    included.append(rf.path)
            continue
        parts.append(block)
        used += cost
        included.append(rf.path)

    if not included:
        # at least return the digest if we built one
        text = "".join(parts) if any("REPOSITORY MAP" in p for p in parts) else ""
        return ContextResult(text=text, tokens=used if text else 0,
                             ranked=ranked)
    return ContextResult(text="".join(parts), files=included, tokens=used,
                         truncated=truncated, ranked=ranked)


__all__ = [
    "RepoDigest", "build_repo_digest", "compress_source",
    "rank_with_recency", "build_context_v2",
]
