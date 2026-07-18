"""MCP (Model Context Protocol) tool surface — Architecture.md §"MCP".

All tools — built-in and user-installed — are exposed through MCP.
This module provides the registry, permission gating, and a small
JSON-RPC adapter the route layer fronts as REST.

Modules:
    registry   — in-memory list of installed servers + their tools
    transport  — stdio/process adapter (subprocess JSON-RPC)
    permissions— per-tool allow/deny + prompt-once model

The full MCP standard (server-side server, tool annotations,
roots, sampling) is out of scope for the initial scaffold —
TODOs flag the gaps. The public surface is stable so the UI's
"Tools" screen and the agent layer can integrate today.
"""
from .registry import (
    Tool,
    Server,
    ToolPermission,
    registry,
)
from .permissions import PermissionStore, default_permission_store
from .dispatcher import invoke, PermissionDeniedError
from .in_process import (
    register_in_process,
    unregister_in_process,
    is_in_process,
    reset_in_process,
)


__all__ = [
    "Tool",
    "Server",
    "ToolPermission",
    "registry",
    "PermissionStore",
    "default_permission_store",
    "invoke",
    "PermissionDeniedError",
    "register_in_process",
    "unregister_in_process",
    "is_in_process",
    "reset_in_process",
]
