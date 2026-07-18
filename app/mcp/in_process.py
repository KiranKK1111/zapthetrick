"""In-process MCP tools — register a plain Python function as a tool (P2-9).

The Agent SDK lets you expose in-process Python functions as MCP tools (no
subprocess, no transport). This gives us the same: `register_in_process("name",
fn, ...)` makes `fn` callable by the agent loop exactly like any other MCP tool.
The function takes the call args dict and returns a JSON-able result (sync or
async).

Mechanics: registered tools are surfaced through the normal
`registry.list_tools()` under a virtual, always-installed "in-process" server,
and `dispatcher.invoke` runs the Python function directly (no transport,
first-party → no permission prompt). The registry starts EMPTY, so this changes
nothing until a host registers a tool.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .registry import Server, Tool, registry

IN_PROCESS_SERVER = "in-process"

# name -> callable(args: dict) -> result | Awaitable[result]
_FNS: dict[str, Callable[[dict], Any | Awaitable[Any]]] = {}


def register_in_process(
    name: str,
    fn: Callable[[dict], Any],
    *,
    description: str = "",
    input_schema: dict | None = None,
    danger: str = "low",
) -> None:
    """Register (or replace) an in-process Python tool the agent can call."""
    if not name or not callable(fn):
        raise ValueError("register_in_process needs a name and a callable")
    _FNS[name] = fn
    srv = registry.get_server(IN_PROCESS_SERVER)
    if srv is None:
        srv = Server(name=IN_PROCESS_SERVER, version="1.0",
                     description="In-process Python tools", installed=True)
    # replace any existing tool of the same name
    srv.tools = [t for t in srv.tools if t.name != name]
    srv.tools.append(Tool(
        name=name, description=description, server=IN_PROCESS_SERVER,
        input_schema=input_schema or {}, danger=danger))
    registry.register_server(srv)


def is_in_process(name: str) -> bool:
    return name in _FNS


async def call_in_process(name: str, args: dict) -> Any:
    """Invoke a registered in-process tool (awaits coroutines)."""
    fn = _FNS.get(name)
    if fn is None:
        raise KeyError(f"in-process tool {name!r} is not registered")
    res = fn(args or {})
    if inspect.isawaitable(res):
        res = await res
    return res


def unregister_in_process(name: str) -> None:
    _FNS.pop(name, None)
    srv = registry.get_server(IN_PROCESS_SERVER)
    if srv is not None:
        srv.tools = [t for t in srv.tools if t.name != name]
        if not srv.tools:
            registry.remove_server(IN_PROCESS_SERVER)
        else:
            registry.register_server(srv)


def reset_in_process() -> None:
    _FNS.clear()
    registry.remove_server(IN_PROCESS_SERVER)


__all__ = [
    "register_in_process", "is_in_process", "call_in_process",
    "unregister_in_process", "reset_in_process", "IN_PROCESS_SERVER",
]
