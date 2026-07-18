"""User hooks (Claude Code-style) for Agent Mode.

Drop executable scripts in `~/.zapthetrick/hooks/<event>/`:
    pretooluse/   — run BEFORE each tool; can DENY (exit≠0, or stdout JSON
                    {"permissionDecision":"deny","permissionDecisionReason":"…"})
    posttooluse/  — run AFTER a tool (informational; output appended as context)
    stop/         — run when the run ends

Each script receives the event JSON on stdin: {event, tool, args, result?}.
`.py` runs with python, `.sh` with bash, `.ps1` with powershell, `.bat`/`.cmd`
directly. Best-effort: a missing dir / failing hook never breaks the run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess

log = logging.getLogger(__name__)

_ROOT = os.path.join(
    os.environ.get("ZAPTHETRICK_HOME") or os.path.expanduser("~/.zapthetrick"),
    "hooks",
)
_TIMEOUT = 20


def _scripts(event: str) -> list[str]:
    d = os.path.join(_ROOT, event)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith((".py", ".sh", ".ps1", ".bat", ".cmd"))
    )


def _cmd(path: str) -> list[str] | None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return ["python", path]
    if ext == ".sh":
        bash = shutil.which("bash")
        return [bash, path] if bash else None
    if ext == ".ps1":
        ps = shutil.which("powershell") or shutil.which("pwsh")
        return [ps, "-NoProfile", "-File", path] if ps else None
    if ext in (".bat", ".cmd"):
        return [path]
    return None


async def _run(path: str, payload: dict) -> tuple[int, str]:
    cmd = _cmd(path)
    if cmd is None:
        return 0, ""

    def _go() -> tuple[int, str]:
        try:
            r = subprocess.run(
                cmd, input=json.dumps(payload), capture_output=True, text=True,
                timeout=_TIMEOUT, errors="replace")
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except Exception as exc:  # noqa: BLE001
            log.info("hook %s failed: %s", os.path.basename(path), exc)
            return 0, ""  # a broken hook never blocks

    return await asyncio.to_thread(_go)


async def run_pre(tool: str, args: dict) -> tuple[bool, str]:
    """(allowed, reason). A hook denies via non-zero exit OR a stdout JSON with
    permissionDecision == 'deny'."""
    for s in _scripts("pretooluse"):
        rc, out = await _run(s, {"event": "PreToolUse", "tool": tool, "args": args})
        try:
            obj = json.loads(out.strip()) if out.strip().startswith("{") else {}
        except Exception:  # noqa: BLE001
            obj = {}
        decision = str(obj.get("permissionDecision", "")).lower()
        if rc != 0 or decision == "deny":
            reason = obj.get("permissionDecisionReason") or \
                f"denied by hook {os.path.basename(s)}"
            return False, str(reason)
    return True, ""


async def run_post(tool: str, args: dict, result: str) -> str:
    """Concatenated extra context from PostToolUse hooks (or '')."""
    extra: list[str] = []
    for s in _scripts("posttooluse"):
        _, out = await _run(
            s, {"event": "PostToolUse", "tool": tool, "args": args,
                "result": result[:4000]})
        if out.strip():
            extra.append(out.strip())
    return "\n".join(extra)


async def run_stop() -> None:
    for s in _scripts("stop"):
        await _run(s, {"event": "Stop"})


def has_hooks() -> bool:
    return any(_scripts(e) for e in ("pretooluse", "posttooluse", "stop"))


__all__ = ["run_pre", "run_post", "run_stop", "has_hooks"]
