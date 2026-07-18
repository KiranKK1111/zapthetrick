"""In-process registry of installed MCP servers and their tools.

This is a *catalog* — it stores metadata (name, version, transport,
tool list) but doesn't itself execute tool calls. The transport
module handles that.

Lifecycle:
    1. App startup: registry.bootstrap() reads `cfg.mcp.servers`
       and instantiates a [Server] for each entry.
    2. Each [Server] is asked for its tool list via the transport.
    3. The UI's Tools screen reads `registry.list_tools()` to render.
    4. Agent calls resolve names to (server, tool) via `registry.get()`.

State is in-process — survives across requests but not restarts.
That's intentional: MCP servers are external processes; we don't
own their state.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)


@dataclass
class Tool:
    """A single tool advertised by an MCP server."""
    name: str
    description: str = ""
    server: str = ""
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    danger: str = "low"   # low | medium | high — drives the prompt-once UX


@dataclass
class ToolPermission:
    """Whether the user has approved this tool."""
    granted: bool = False
    granted_at_ms: int | None = None
    rationale: str = ""


@dataclass
class Server:
    """An installed MCP server. `transport` carries the launch params.

    `installed` is True after the user accepted the install prompt;
    False for a discovered-but-not-installed entry (e.g. from a
    public catalog). The Tools screen distinguishes the two states.
    """
    name: str
    version: str = ""
    description: str = ""
    transport: dict = field(default_factory=dict)   # {kind: 'stdio', cmd: [...]}
    installed: bool = False
    tools: list[Tool] = field(default_factory=list)
    healthy: bool = True
    last_error: str | None = None


class Registry:
    """Thread-safe in-memory registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._servers: dict[str, Server] = {}

    def register_server(self, server: Server) -> None:
        with self._lock:
            self._servers[server.name] = server

    def list_servers(self) -> list[Server]:
        with self._lock:
            return list(self._servers.values())

    def get_server(self, name: str) -> Server | None:
        with self._lock:
            return self._servers.get(name)

    def remove_server(self, name: str) -> bool:
        with self._lock:
            return self._servers.pop(name, None) is not None

    def list_tools(self) -> list[Tool]:
        with self._lock:
            out: list[Tool] = []
            for s in self._servers.values():
                if not s.installed:
                    continue
                out.extend(s.tools)
            return out

    def find_tool(self, name: str) -> tuple[Server, Tool] | None:
        with self._lock:
            for s in self._servers.values():
                if not s.installed:
                    continue
                for t in s.tools:
                    if t.name == name:
                        return s, t
        return None

    def bootstrap_from_config(self, config: list[dict]) -> int:
        """Wire up the servers declared in `cfg.mcp.servers`. Returns
        the count loaded."""
        if not config:
            return 0
        loaded = 0
        for entry in config:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            srv = Server(
                name=entry["name"],
                version=entry.get("version", ""),
                description=entry.get("description", ""),
                transport=entry.get("transport") or {},
                installed=bool(entry.get("installed", True)),
            )
            # TODO: probe the server over its transport for live tools.
            # The scaffold below trusts the config-declared tools.
            for t in entry.get("tools") or []:
                if isinstance(t, dict) and t.get("name"):
                    srv.tools.append(
                        Tool(
                            name=t["name"],
                            description=t.get("description", ""),
                            server=srv.name,
                            input_schema=t.get("input_schema") or {},
                            output_schema=t.get("output_schema") or {},
                            danger=t.get("danger", "low"),
                        )
                    )
            self.register_server(srv)
            loaded += 1
        log.info("mcp: loaded %d server(s) from config", loaded)
        return loaded

    def reset_for_tests(self) -> None:
        with self._lock:
            self._servers.clear()


registry = Registry()


__all__ = ["Tool", "Server", "ToolPermission", "Registry", "registry"]
