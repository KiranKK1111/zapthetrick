"""Context engineering for the chat agent loop (Phase 5, report #1/#36).

A real uploaded repo is far bigger than a free model's context window. Dumping
the whole tree (or letting the agent blindly read files) wastes the budget and
derails weak models. This module RANKS the workspace's files by relevance to
the task and packs the highest-value ones into a token-bounded "relevant
project context" preamble the agent sees up front — so it starts on the right
files and stays within the window.

Deterministic + offline (no LLM, no network): ranking combines
  • identifier / path overlap with the task,
  • code-graph centrality (hub files with many callers/callees),
  • entrypoint + task-keyword heuristics (main/app/index/server, test/config).
Then it packs symbol outlines + head excerpts until the budget is hit.

    res = build_context(workspace, task)         # ContextResult
    res.text         # the preamble to inject (empty if nothing useful)
    res.files        # included paths, ranked
    res.tokens       # estimated tokens used
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.codegraph.builder import _SKIP_DIRS, _is_source

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_ENTRYPOINTS = {
    "main", "app", "index", "server", "__init__", "cli", "manage", "wsgi",
    "asgi", "application", "program", "mod", "lib", "run",
}
_MAX_FILE_BYTES = 1_500_000
_MAX_WALK_FILES = 4_000


def _est_tokens(text: str) -> int:
    """Rough input-token estimate (chars/4) — matches the engine's estimator."""
    return max(1, len(text or "") // 4)


@dataclass
class ContextBudget:
    max_tokens: int = 6000      # token budget for the injected code context
    max_files: int = 40         # never rank/include more than this many files
    head_lines: int = 50        # excerpt size (lines) per included file
    outline_only_after: int = 8  # after N excerpted files, include outlines only


@dataclass
class RankedFile:
    path: str
    score: float
    reason: str


@dataclass
class ContextResult:
    text: str = ""
    files: list[str] = field(default_factory=list)
    tokens: int = 0
    truncated: bool = False
    ranked: list[RankedFile] = field(default_factory=list)


# --------------------------------------------------------------------------
# Walk + read the workspace's source files (bounded).
# --------------------------------------------------------------------------
def _read_workspace(workspace: str) -> list[tuple[str, str]]:
    root = os.path.realpath(workspace)
    out: list[tuple[str, str]] = []
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
                if os.path.getsize(full) > _MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    out.append((rel, f.read()))
            except OSError:
                continue
            if len(out) >= _MAX_WALK_FILES:
                return out
    return out


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def _task_tokens(task: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(task or "")}


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def rank_files(
    files: list[tuple[str, str]],
    task: str,
    *,
    limit: int = 40,
) -> list[RankedFile]:
    """Score every source file for relevance to `task`. Pure + deterministic."""
    if not files:
        return []
    tokens = _task_tokens(task)

    # Optional code-graph centrality (best-effort — never blocks ranking).
    centrality: dict[str, int] = {}
    symbols_by_file: dict[str, set[str]] = {}
    try:
        from app.codegraph.builder import build_code_graph
        g = build_code_graph(files)
        for n in g.nodes.values():
            if n.kind == "file":
                continue
            symbols_by_file.setdefault(n.path, set()).add(n.name.lower())
            deg = len(g.in_edges(n.id, "calls")) + len(g.out_edges(n.id, "calls"))
            centrality[n.path] = centrality.get(n.path, 0) + deg
    except Exception:  # noqa: BLE001 — ranking still works on heuristics alone
        pass

    ranked: list[RankedFile] = []
    for path, source in files:
        score = 0.0
        reasons: list[str] = []
        stem = _stem(path)
        path_l = path.lower()

        # 1) Task-identifier overlap with the file's own symbols / path.
        syms = symbols_by_file.get(path, set())
        sym_hits = tokens & syms
        if sym_hits:
            score += 5.0 * len(sym_hits)
            reasons.append("mentions " + ", ".join(sorted(sym_hits)[:4]))
        path_hits = {t for t in tokens if len(t) > 2 and t in path_l}
        if path_hits:
            score += 3.0 * len(path_hits)
            reasons.append("path matches " + ", ".join(sorted(path_hits)[:3]))
        if stem in tokens:
            score += 4.0
            reasons.append("filename named in task")

        # 2) Graph centrality (hub files matter even without a direct mention).
        deg = centrality.get(path, 0)
        if deg:
            score += min(4.0, deg * 0.25)
            if deg >= 4:
                reasons.append("central (many calls)")

        # 3) Entrypoint / structural heuristics.
        if stem in _ENTRYPOINTS:
            score += 2.5
            reasons.append("entrypoint")
        if any(k in path_l for k in ("readme", "config", "settings",
                                     "requirements", "package.json",
                                     "pyproject", "dockerfile", "makefile")):
            score += 1.0

        # 4) Task-keyword → file-kind affinity.
        if ({"test", "tests", "testing", "unit"} & tokens) and \
                ("test" in path_l or "spec" in path_l):
            score += 3.0
            reasons.append("test file (task is about tests)")
        if ({"api", "route", "routes", "endpoint", "endpoints"} & tokens) and \
                any(k in path_l for k in ("route", "api", "controller", "view")):
            score += 2.5
            reasons.append("api/route file")

        # Tiny prior so smaller/leaf files don't all tie at 0 — shorter files
        # are cheaper to include, break ties toward them.
        score += max(0.0, 0.5 - len(source) / 200_000.0)

        if score > 0:
            ranked.append(RankedFile(path, score, "; ".join(reasons) or "related"))

    ranked.sort(key=lambda r: (-r.score, r.path))
    return ranked[:limit]


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------
def _outline(source: str, max_syms: int = 24) -> str:
    """A cheap signature outline: def/class/func/interface/fn lines."""
    pat = re.compile(
        r"^\s*(?:export\s+|public\s+|private\s+|async\s+)*"
        r"(?:def|class|func|function|interface|struct|trait|enum|type|impl|"
        r"fn|module|namespace)\b.*$",
        re.M,
    )
    lines = [m.group(0).strip() for m in pat.finditer(source)]
    if not lines:
        return ""
    return "\n".join(lines[:max_syms])


def build_context(
    workspace: str,
    task: str,
    *,
    budget: ContextBudget | None = None,
) -> ContextResult:
    """Rank the workspace's files for `task` and pack a token-bounded preamble.
    Returns an empty result when there's nothing useful (empty/fresh build)."""
    budget = budget or ContextBudget()
    files = _read_workspace(workspace)
    if not files:
        return ContextResult()
    by_path = dict(files)
    ranked = rank_files(files, task, limit=budget.max_files)
    if not ranked:
        return ContextResult()

    header = (
        "RELEVANT PROJECT CONTEXT (ranked for THIS task — use it to go straight "
        "to the right files; read a file in full before editing it):\n"
    )
    parts: list[str] = [header]
    used = _est_tokens(header)
    included: list[str] = []
    truncated = False

    for i, rf in enumerate(ranked):
        source = by_path.get(rf.path, "")
        if i < budget.outline_only_after:
            body_lines = source.splitlines()[: budget.head_lines]
            body = "\n".join(body_lines)
            if len(body_lines) < len(source.splitlines()):
                body += "\n…"
            block = f"\n### {rf.path}  — {rf.reason}\n```\n{body}\n```\n"
        else:
            outline = _outline(source)
            if not outline:
                continue
            block = f"\n### {rf.path}  — {rf.reason} (outline)\n```\n{outline}\n```\n"

        cost = _est_tokens(block)
        if used + cost > budget.max_tokens:
            truncated = True
            # Try a cheap outline instead of skipping outright.
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
        return ContextResult(ranked=ranked)
    return ContextResult(
        text="".join(parts), files=included, tokens=used,
        truncated=truncated, ranked=ranked,
    )


__all__ = [
    "ContextBudget", "ContextResult", "RankedFile",
    "rank_files", "build_context",
]
