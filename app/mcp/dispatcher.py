"""Resolve tool name → invoke via transport → return result.

Public surface:
    invoke(tool_name, args) -> dict   raises [PermissionError] if not granted

Permission gate runs first. Architecture.md commits to "prompt-once
for medium-danger, allow-by-default for low-danger, prompt-every-time
for high-danger". The first two policies are honoured here; high-
danger tools need a per-call confirmation hook on the route layer.
"""
from __future__ import annotations

import logging
from typing import Any

from .permissions import default_permission_store
from .registry import registry
from .transport import MCPError, get_transport


log = logging.getLogger(__name__)


class PermissionDeniedError(RuntimeError):
    """Raised when the user hasn't granted access to this tool."""


async def invoke(tool_name: str, args: dict[str, Any]) -> dict:
    """Run an MCP tool. Returns the JSON-RPC result on success.

    Raises:
      KeyError              if the tool isn't installed
      PermissionDeniedError if it isn't granted
      MCPError              on transport / protocol failure
    """
    pair = registry.find_tool(tool_name)
    if pair is None:
        raise KeyError(f"tool {tool_name!r} is not installed")
    server, tool = pair

    # In-process tools (P2-9): first-party Python functions — run them directly,
    # no transport, no permission prompt.
    from .in_process import call_in_process, is_in_process

    if is_in_process(tool_name):
        log.info("mcp dispatch (in-process): %s(%s)", tool_name,
                 list(args.keys()))
        result = await call_in_process(tool_name, args)
        return result if isinstance(result, dict) else {"result": result}

    # Permission gate.
    store = default_permission_store()
    if not store.is_granted(tool_name):
        # Architecture.md: auto-grant low-danger if cfg flag is on.
        # We import lazily to avoid pulling config into permissions.py.
        from app.core.config_loader import cfg

        if tool.danger == "low" and cfg.mcp.auto_grant_low_danger:
            store.grant(tool_name, rationale="auto-grant: low danger")
        else:
            raise PermissionDeniedError(
                f"tool {tool_name!r} is not granted (danger={tool.danger})"
            )

    # Transport.
    transport_cfg = server.transport or {}
    if transport_cfg.get("kind") not in (None, "stdio"):
        raise MCPError(
            f"transport {transport_cfg.get('kind')!r} not implemented (stdio only)"
        )
    cmd = transport_cfg.get("cmd")
    if not cmd or not isinstance(cmd, list):
        raise MCPError(f"server {server.name!r} has no `cmd` in transport")

    transport = await get_transport(server.name, cmd, transport_cfg.get("env"))
    if not transport.alive:
        raise MCPError(f"server {server.name!r} failed to start")

    log.info("mcp dispatch: %s.%s(%s)", server.name, tool_name, list(args.keys()))
    result = await transport.call("tools/call", {"name": tool_name, "arguments": args})
    return result if isinstance(result, dict) else {"result": result}


__all__ = ["invoke", "PermissionDeniedError"]
