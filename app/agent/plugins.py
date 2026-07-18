"""Plugin loader (Phase 4) — Claude Code `.claude-plugin/plugin.json` bundles.

A plugin is a folder under `~/.zapthetrick/plugins/<name>/` with:
    .claude-plugin/plugin.json   (name, version, description)
    commands/*.md                (slash commands — merged into the registry)
    agents/*.md                  (named subagent prompts — for the Task tool)
    skills/<name>/SKILL.md        (skills)
    .mcp.json                    (MCP servers)

Everything auto-discovers; no registration. The antigravity bundles already fit
this shape, and the official claude-plugins-* repos drop in the same way.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

_DIR = os.path.join(
    os.environ.get("ZAPTHETRICK_HOME") or os.path.expanduser("~/.zapthetrick"),
    "plugins",
)


def _count(root: str, sub: str, ext: str) -> int:
    d = os.path.join(root, sub)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith(ext))


def _count_skills(root: str) -> int:
    d = os.path.join(root, "skills")
    if not os.path.isdir(d):
        return 0
    return sum(
        1 for s in os.listdir(d)
        if os.path.isfile(os.path.join(d, s, "SKILL.md"))
    )


@lru_cache(maxsize=1)
def list_plugins() -> list[dict]:
    out: list[dict] = []
    if not os.path.isdir(_DIR):
        return out
    for name in sorted(os.listdir(_DIR)):
        root = os.path.join(_DIR, name)
        manifest = os.path.join(root, ".claude-plugin", "plugin.json")
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.info("plugin %s manifest unreadable: %s", name, exc)
            continue
        out.append({
            "id": name,
            "name": str(meta.get("name", name)),
            "version": str(meta.get("version", "")),
            "description": str(meta.get("description", "")),
            "root": root,
            "commands": _count(root, "commands", ".md"),
            "agents": _count(root, "agents", ".md"),
            "skills": _count_skills(root),
            "hooks": _count(root, "hooks", ".md") + _count(root, "hooks", ".sh"),
            "mcp": os.path.isfile(os.path.join(root, ".mcp.json")),
        })
    return out


def plugin_command_dirs() -> list[str]:
    """`<plugin>/commands` dirs — folded into the slash-command registry."""
    return [
        os.path.join(p["root"], "commands")
        for p in list_plugins()
        if os.path.isdir(os.path.join(p["root"], "commands"))
    ]


def plugin_agents() -> dict[str, str]:
    """name → system-prompt body, from every plugin's `agents/*.md`. Available to
    the Task tool as named subagents."""
    agents: dict[str, str] = {}
    for p in list_plugins():
        adir = os.path.join(p["root"], "agents")
        if not os.path.isdir(adir):
            continue
        for fn in sorted(os.listdir(adir)):
            if not fn.lower().endswith(".md"):
                continue
            try:
                with open(os.path.join(adir, fn), encoding="utf-8") as f:
                    agents[os.path.splitext(fn)[0]] = f.read()
            except Exception:  # noqa: BLE001
                continue
    return agents


def reload() -> None:
    list_plugins.cache_clear()


__all__ = ["list_plugins", "plugin_command_dirs", "plugin_agents", "reload"]
