"""REST surface over the MCP registry and permission store.

Endpoints:
    GET    /api/mcp/servers              list installed + discovered servers
    POST   /api/mcp/servers              install a new server (config-only)
    DELETE /api/mcp/servers/{name}       uninstall
    GET    /api/mcp/tools                flat list of every tool from installed servers
    POST   /api/mcp/tools/{name}/grant   approve a tool for use
    POST   /api/mcp/tools/{name}/revoke  drop a previously-granted approval
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.mcp import registry
from app.mcp.permissions import default_permission_store
from app.mcp.registry import Server


router = APIRouter(prefix="/api/mcp")


class GrantBody(BaseModel):
    rationale: str = ""


@router.get("/servers")
async def list_servers() -> list[dict]:
    return [
        {
            "name": s.name,
            "version": s.version,
            "description": s.description,
            "installed": s.installed,
            "transport": s.transport,
            "healthy": s.healthy,
            "tools": [{"name": t.name, "danger": t.danger} for t in s.tools],
        }
        for s in registry.list_servers()
    ]


@router.post("/servers")
async def install_server(body: dict) -> dict:
    """Add a server to the registry. Validates only minimal shape —
    the transport launcher does its own validation when first invoked."""
    name = (body or {}).get("name")
    if not name:
        raise HTTPException(status_code=400, detail="`name` is required")
    srv = Server(
        name=str(name),
        version=str(body.get("version") or ""),
        description=str(body.get("description") or ""),
        transport=body.get("transport") or {},
        installed=True,
    )
    for t in body.get("tools") or []:
        # Re-use registry's loader to keep shape consistent.
        pass  # tool list comes from a `discover` round-trip — TODO
    registry.register_server(srv)
    return {"ok": True, "name": srv.name}


@router.delete("/servers/{name}")
async def uninstall_server(name: str) -> dict:
    ok = registry.remove_server(name)
    if not ok:
        raise HTTPException(status_code=404, detail="server not installed")
    return {"ok": True}


@router.get("/tools")
async def list_tools() -> list[dict]:
    store = default_permission_store()
    return [
        {
            "name": t.name,
            "server": t.server,
            "description": t.description,
            "danger": t.danger,
            "granted": store.is_granted(t.name),
        }
        for t in registry.list_tools()
    ]


@router.post("/tools/{name}/grant")
async def grant_tool(name: str, body: GrantBody) -> dict:
    if registry.find_tool(name) is None:
        raise HTTPException(status_code=404, detail="tool not installed")
    default_permission_store().grant(name, body.rationale)
    return {"ok": True, "granted": True}


@router.post("/tools/{name}/revoke")
async def revoke_tool(name: str) -> dict:
    existed = default_permission_store().revoke(name)
    return {"ok": True, "existed": existed}


class InvokeBody(BaseModel):
    arguments: dict = {}


@router.post("/tools/{name}/invoke")
async def invoke_tool(name: str, body: InvokeBody) -> dict:
    """Run a granted tool. Returns the JSON-RPC result envelope."""
    from app.mcp import invoke as _invoke, PermissionDeniedError
    from app.mcp.transport import MCPError

    try:
        result = await _invoke(name, body.arguments)
    except KeyError:
        raise HTTPException(404, detail=f"tool {name!r} not installed")
    except PermissionDeniedError as exc:
        raise HTTPException(403, detail=str(exc))
    except MCPError as exc:
        raise HTTPException(502, detail=str(exc))
    return {"ok": True, "result": result}
