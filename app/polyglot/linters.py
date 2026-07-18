"""Fast lint pass for a code snippet — best-effort, fail-open (fills the
`linters.py — TODO` from __init__).

Returns a list of findings (line, message, code, severity). When the linter
binary isn't installed the result is [] — a missing toolchain degrades to "not
linted", never an error. Mirrors `formatters.py`: subprocess, no Python bindings.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class LintFinding:
    line: int
    message: str
    code: str = ""
    severity: str = "warning"    # error | warning | info

    def to_dict(self) -> dict:
        return {"line": self.line, "message": self.message,
                "code": self.code, "severity": self.severity}


async def _run(argv: list[str], code: str, timeout_s: float):
    """(returncode, stdout, stderr) or None when the tool can't run."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(
            proc.communicate(input=code.encode("utf-8")), timeout=timeout_s)
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
        log.info("linter %s skipped: %s", argv[0] if argv else "?", exc)
        return None


async def _ruff(code: str, timeout_s: float) -> list[LintFinding]:
    if shutil.which("ruff") is None:
        return []
    res = await _run(["ruff", "check", "--output-format", "json",
                      "--stdin-filename", "snippet.py", "-"], code, timeout_s)
    if not res:
        return []
    _, out, _ = res
    try:
        data = json.loads(out or "[]")
    except Exception:  # noqa: BLE001
        return []
    findings: list[LintFinding] = []
    for d in data if isinstance(data, list) else []:
        loc = d.get("location") or {}
        findings.append(LintFinding(
            line=int(loc.get("row", 0) or 0),
            message=str(d.get("message", "")),
            code=str(d.get("code") or ""),
            severity="warning"))
    return findings[:50]


async def _eslint(code: str, lang: str, timeout_s: float) -> list[LintFinding]:
    if shutil.which("eslint") is None:
        return []
    fname = "snippet.ts" if lang == "typescript" else "snippet.js"
    res = await _run(["eslint", "--stdin", "--stdin-filename", fname,
                      "--format", "json"], code, timeout_s)
    if not res:
        return []
    _, out, _ = res
    try:
        data = json.loads(out or "[]")
    except Exception:  # noqa: BLE001
        return []
    findings: list[LintFinding] = []
    for file_res in data if isinstance(data, list) else []:
        for m in (file_res.get("messages") or []):
            findings.append(LintFinding(
                line=int(m.get("line", 0) or 0),
                message=str(m.get("message", "")),
                code=str(m.get("ruleId") or ""),
                severity="error" if m.get("severity") == 2 else "warning"))
    return findings[:50]


async def lint_code(language: str, code: str, *,
                    timeout_s: float = 6.0) -> list[LintFinding]:
    """Lint `code`. Returns [] when no linter is available for the language."""
    lang = (language or "").strip().lower()
    if not (code or "").strip():
        return []
    if lang == "python":
        return await _ruff(code, timeout_s)
    if lang in ("javascript", "typescript"):
        return await _eslint(code, lang, timeout_s)
    return []


async def fix_code(language: str, code: str, *,
                   timeout_s: float = 8.0) -> str:
    """Auto-FIX safe lint issues (ruff --fix / eslint --fix). Returns the
    corrected source, or the ORIGINAL unchanged when no fixer is available or
    it produced nothing. Never raises."""
    lang = (language or "").strip().lower()
    src = code or ""
    if not src.strip():
        return src
    try:
        if lang == "python" and shutil.which("ruff") is not None:
            res = await _run(["ruff", "check", "--fix", "--exit-zero",
                              "--stdin-filename", "snippet.py", "-"], src, timeout_s)
            if res:
                _, out, _ = res
                return out if out.strip() else src
        elif lang in ("javascript", "typescript") and shutil.which("eslint") is not None:
            fname = "snippet.ts" if lang == "typescript" else "snippet.js"
            res = await _run(["eslint", "--stdin", "--stdin-filename", fname,
                              "--fix-dry-run", "--format", "json"], src, timeout_s)
            if res:
                _, out, _ = res
                try:
                    data = json.loads(out or "[]")
                    fixed = (data[0].get("output") if data else None)
                    if isinstance(fixed, str) and fixed.strip():
                        return fixed
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    return src


__all__ = ["LintFinding", "lint_code", "fix_code"]
