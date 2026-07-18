"""Permission model for Agent Mode (Phase 3).

Modes (mirroring Claude Code):
- plan         — READ-ONLY: no write/edit/bash. Safe exploration & planning.
- acceptEdits  — auto-approve reads + file edits; bash is allowed unless it hits
                 the hard deny-list.
- auto         — approve everything except the hard deny-list (power mode).
- ask          — EVERY action (incl. reads/searches) requires explicit approval
                 (the loop emits an `approval` event; until a FE/WS answers,
                 callers pass an approver callback — defaults to deny for
                 unattended runs).

A hard DENY-LIST blocks catastrophic bash regardless of mode (rm -rf /, mkfs,
disk wipes, fork bombs, shutdown, piping the internet into a shell, …).
"""
from __future__ import annotations

import re

from app.agent.tools import SPEC_BY_NAME

MODES = ("plan", "acceptEdits", "auto", "ask")

# Claude Code's mode vocabulary → our internal names, so callers can use either:
#   default            → ask   (prompt before each tool)
#   acceptEdits        → acceptEdits
#   plan               → plan  (read-only)
#   bypassPermissions  → auto  (no prompts)
_MODE_ALIASES = {
    "default": "ask",
    "bypasspermissions": "auto",
    "bypass": "auto",
    "accept_edits": "acceptEdits",
    "acceptedits": "acceptEdits",
    "read-only": "plan",
    "readonly": "plan",
}


def normalize_mode(mode: str | None, *, default: str = "acceptEdits") -> str:
    """Map any accepted mode spelling (incl. Claude's names) to an internal
    MODE; unknown values fall back to `default`."""
    m = (mode or "").strip()
    if m in MODES:
        return m
    alias = _MODE_ALIASES.get(m.lower())
    if alias:
        return alias
    return default

# Catastrophic / irreversible-at-scale commands — always denied.
_DENY = [
    r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|\.\.)(\s|$)",  # rm -rf / ~ ..
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;",                          # fork bomb
    r"\bmkfs\b", r"\bdd\b[^|]*\bof=/dev/", r">\s*/dev/sd",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\bformat\s+[a-z]:", r"Remove-Item[^\n]*-Recurse[^\n]*-Force[^\n]*[\\/]",
    r"\bgit\b[^\n]*\bpush\b[^\n]*--force",                   # force-push
    r"curl[^\n|]*\|\s*(sh|bash|python)", r"wget[^\n|]*\|\s*(sh|bash)",
    r"\bsudo\b", r"\bchmod\s+-R\s+777\s+/",
]
_DENY_RE = [re.compile(p, re.IGNORECASE) for p in _DENY]


def deny_reason(command: str) -> str | None:
    for rx in _DENY_RE:
        if rx.search(command or ""):
            return "blocked by the agent safety deny-list (destructive command)"
    return None


def decide(tool: str, args: dict, mode: str) -> tuple[str, str]:
    """Return (decision, reason): decision ∈ {'allow','deny','ask'}."""
    mode = normalize_mode(mode)
    spec = SPEC_BY_NAME.get(tool)
    if spec is None:
        return "deny", f"unknown tool '{tool}'"

    if tool == "bash":
        why = deny_reason(str(args.get("command", "")))
        if why:
            return "deny", why

    if mode == "plan" and (spec.writes or spec.runs):
        return "deny", "plan mode is read-only (no writes or commands)"

    if mode == "auto":
        return "allow", ""

    if mode == "ask":
        # Ask before EVERY action — INCLUDING reads/searches — so the user can
        # vet what the agent looks at (e.g. before it reads other folders/files).
        return "ask", "needs approval"

    # acceptEdits (default): reads/edits auto; bash allowed (deny-list already
    # applied), so it's 'allow' here too.
    return "allow", ""


__all__ = ["MODES", "decide", "deny_reason", "normalize_mode"]
