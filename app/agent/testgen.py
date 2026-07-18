"""Test-first rigor (P2-5, report_2 §P2-5).

Claude is rigorous about tests: it writes tests for new/changed code, runs them,
and won't claim done while they fail. This module gives the agent the same
discipline on free models by making the TEST SURFACE explicit + measurable:

  • changed_symbols()  — what symbols were added / changed / removed since the
        baseline (reuses the AST semantic-diff per changed file).
  • test_surface()      — folds that into a structured report: which added/
        changed symbols are NOT referenced by any test file (the untested
        surface), whether the project even has a test system, plus counts.
  • TEST_FIRST_DIRECTIVE — the system steer injected into build/edit runs:
        write characterization tests before a refactor, add tests for new/
        changed symbols, and verify they pass before finishing.

The pure helpers (`is_test_file`, `untested_symbols`, `leaf_name`) are offline +
deterministic; `changed_symbols`/`test_surface` do the git+fs IO. Test
*generation* itself is the model's job (guided by this surface); we make the
target unambiguous and feed coverage into the confidence band.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_TEST_RE = re.compile(
    r"(^|/)(tests?|spec|specs|__tests__)(/|$)"
    r"|(^|/)test_[^/]+$|_test\.[a-z]+$|\.test\.[a-z]+$|\.spec\.[a-z]+$"
    r"|Test[A-Z][^/]*\.[a-z]+$|[^/]+Test\.[a-z]+$|[^/]+Tests\.[a-z]+$",
    re.IGNORECASE,
)

TEST_FIRST_DIRECTIVE = (
    "TEST-FIRST RIGOR (do this as you work):\n"
    "- For every NEW or CHANGED function/class, add or update a focused test "
    "for it in the project's test suite.\n"
    "- Before REFACTORING existing behavior, first write a characterization "
    "test that captures the current behavior, so you can prove the refactor "
    "didn't change it.\n"
    "- Run the tests (`verify`, or the project's test command via `bash`) and "
    "make sure they PASS before you call `final`. Don't claim success you "
    "haven't verified."
)


def is_test_file(path: str) -> bool:
    """Heuristic: does `path` look like a test/spec file?"""
    p = (path or "").replace("\\", "/")
    return bool(_TEST_RE.search(p))


def leaf_name(qualified_name: str) -> str:
    """The last component of a qualified symbol name (e.g. `User.save` → save)."""
    q = (qualified_name or "").strip()
    if not q:
        return ""
    return q.split(".")[-1].split("::")[-1]


def untested_symbols(symbols: list[str], test_blobs: list[str]) -> list[str]:
    """Of `symbols` (qualified names), those NOT referenced in any test blob.

    Pure + deterministic. A symbol counts as covered when its leaf name appears
    as a word in any test file's content (a deliberately lenient proxy — better
    to under-flag than to nag about a symbol that clearly has a test)."""
    if not symbols:
        return []
    haystack = "\n".join(test_blobs)
    if not haystack.strip():
        return list(symbols)
    out: list[str] = []
    for q in symbols:
        leaf = leaf_name(q)
        if not leaf:
            continue
        if re.search(rf"\b{re.escape(leaf)}\b", haystack):
            continue
        out.append(q)
    return out


@dataclass
class TestSurface:
    added: list[str] = field(default_factory=list)        # qualified names
    changed: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)     # added/changed w/o tests
    tests_added: int = 0                                  # new symbols in test files
    has_test_system: bool = False
    test_files_changed: int = 0

    @property
    def untested_count(self) -> int:
        return len(self.untested)

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "changed": self.changed,
            "untested": self.untested,
            "tests_added": self.tests_added,
            "has_test_system": self.has_test_system,
            "test_files_changed": self.test_files_changed,
        }

    def summary(self) -> str:
        if not (self.added or self.changed):
            return ""
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} symbol(s) added")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        if self.tests_added:
            parts.append(f"{self.tests_added} new test(s)")
        if self.untested:
            parts.append(f"{len(self.untested)} change(s) without tests: "
                         + ", ".join(self.untested[:6]))
        return "; ".join(parts)


async def changed_symbols(workspace: str, *, max_files: int = 20) -> dict:
    """Per changed code file since baseline: {file: semantic_diff dict}.

    Reuses the staged git diff + the AST semantic-diff. Best-effort → {}."""
    from app.agent_workspace.runner import run_in_workspace
    from app.codegraph.semantic_diff import semantic_diff
    from app.codegraph.tsutil import language_for

    try:
        await run_in_workspace("git add -A", cwd=workspace, timeout=60)
        names_r = await run_in_workspace(
            "git diff --cached --name-only HEAD", cwd=workspace, timeout=60)
    except Exception:  # noqa: BLE001
        return {}
    files = [f.strip() for f in (names_r.stdout or "").splitlines() if f.strip()]
    out: dict[str, dict] = {}
    root = os.path.realpath(workspace)
    for rel in files[:max_files]:
        if language_for(rel) is None:
            continue
        try:
            old_r = await run_in_workspace(
                f'git show "HEAD:{rel}"', cwd=workspace, timeout=30)
            old_src = old_r.stdout if (old_r.ok and not old_r.denied) else ""
        except Exception:  # noqa: BLE001
            old_src = ""
        target = os.path.realpath(os.path.join(root, rel))
        if not (target == root or target.startswith(root + os.sep)) \
                or not os.path.isfile(target):
            continue
        try:
            with open(target, encoding="utf-8", errors="replace") as fh:
                new_src = fh.read()
        except OSError:
            continue
        d = semantic_diff(old_src, new_src, path=rel)
        if d.get("added") or d.get("changed") or d.get("removed"):
            out[rel] = d
    return out


async def test_surface(workspace: str, *, max_files: int = 20) -> TestSurface:
    """Compute the test surface for the workspace's changes since baseline."""
    from app.agent_workspace.buildsys import detect_build_system

    diffs = await changed_symbols(workspace, max_files=max_files)
    surface = TestSurface()
    try:
        bs = detect_build_system(workspace)
        surface.has_test_system = bool(bs and bs.test)
    except Exception:  # noqa: BLE001
        surface.has_test_system = False

    prod_symbols: list[str] = []
    test_blobs: list[str] = []
    root = os.path.realpath(workspace)
    for rel, d in diffs.items():
        if is_test_file(rel):
            surface.test_files_changed += 1
            surface.tests_added += len(d.get("added") or [])
            # collect the test file's content so we can match coverage
            target = os.path.join(root, rel)
            try:
                with open(target, encoding="utf-8", errors="replace") as fh:
                    test_blobs.append(fh.read())
            except OSError:
                pass
            continue
        added = list(d.get("added") or [])
        changed = [c["symbol"] for c in (d.get("changed") or [])
                   if isinstance(c, dict) and c.get("symbol")]
        surface.added.extend(added)
        surface.changed.extend(changed)
        prod_symbols.extend(added + changed)

    # also fold in ALL existing test files so coverage isn't limited to changed
    # ones (a symbol may be covered by a pre-existing test).
    if prod_symbols:
        test_blobs.extend(_all_test_blobs(root))
        surface.untested = untested_symbols(prod_symbols, test_blobs)
    return surface


_MAX_TEST_SCAN = 400


def _all_test_blobs(root: str) -> list[str]:
    """Read the content of every test file in the workspace (bounded)."""
    from app.chat.context_builder import _SKIP_DIRS, _is_source

    out: list[str] = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS]
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/")
            if not _is_source(rel) or not is_test_file(rel):
                continue
            try:
                with open(os.path.join(dp, fn), encoding="utf-8",
                          errors="replace") as fh:
                    out.append(fh.read())
            except OSError:
                continue
            if len(out) >= _MAX_TEST_SCAN:
                return out
    return out


__all__ = [
    "TEST_FIRST_DIRECTIVE", "TestSurface",
    "is_test_file", "leaf_name", "untested_symbols",
    "changed_symbols", "test_surface",
]
