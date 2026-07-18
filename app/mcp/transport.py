"""MCP transport — stdio JSON-RPC subprocess adapter.

Architecture.md §"MCP": the registry stores server metadata, this
module owns the live subprocess + JSON-RPC framing.

MCP's stdio transport is the LSP-style "Content-Length" framing —
one JSON-RPC message per logical block, length-prefixed. We
implement the minimum surface needed to:
    initialize             handshake with capabilities
    tools/list             enumerate the server's tools
    tools/call             invoke a tool with arguments

Higher-level features (sampling, roots, prompts, resources, server
notifications) are TODO. The public surface is stable so we can grow
into them.

Failure model: subprocess crashes / non-zero exits leave the
`StdioTransport.alive` flag at False; callers fall back to "tool
unavailable" cleanly rather than raising.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)


class MCPError(RuntimeError):
    """Raised when the JSON-RPC peer returns an error envelope or the
    transport dies."""


@dataclass
class StdioTransport:
    """Manages one MCP server subprocess.

    `cmd` is the argv to launch (e.g. ["npx", "-y", "@my/mcp-tool"]).
    `env` overrides PATH-relative process env. After `start()` returns
    successfully, `call()` may be invoked repeatedly.
    """
    cmd: list[str]
    env: dict[str, str] | None = None
    name: str = ""
    timeout_s: float = 8.0

    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _seq: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _alive: bool = field(default=False, init=False)

    @property
    def alive(self) -> bool:
        return self._alive and self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if self.alive:
            return
        log.info("mcp transport: launching %s", " ".join(self.cmd))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )
        except (FileNotFoundError, OSError) as exc:
            log.warning("mcp transport: launch failed (%s): %s", self.cmd, exc)
            self._alive = False
            return
        self._alive = True
        # Initialize handshake — defensive timeout.
        try:
            await self.call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "zapthetrick", "version": "0.2"},
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp transport: initialize failed for %s: %s", self.name, exc)
            await self.stop()

    async def stop(self) -> None:
        if self._proc is None:
            self._alive = False
            return
        try:
            if self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
        except Exception:  # noqa: BLE001
            pass
        self._alive = False
        self._proc = None

    async def call(self, method: str, params: dict[str, Any]) -> Any:
        """Send one JSON-RPC request and await its reply.

        Raises [MCPError] on transport failure, non-zero error
        envelope, or timeout.
        """
        if not self.alive:
            raise MCPError(f"server {self.name!r} is not running")
        async with self._lock:
            self._seq += 1
            req_id = self._seq
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            frame = (
                f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data
            )
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(frame)
            await self._proc.stdin.drain()

            try:
                reply = await asyncio.wait_for(
                    self._read_one(), timeout=self.timeout_s
                )
            except asyncio.TimeoutError:
                raise MCPError(f"mcp call {method} timed out after {self.timeout_s}s")

        if not isinstance(reply, dict):
            raise MCPError(f"malformed reply: {reply!r}")
        if "error" in reply and reply["error"]:
            err = reply["error"]
            raise MCPError(str(err.get("message") or err))
        return reply.get("result")

    async def _read_one(self) -> Any:
        """Read one Content-Length-framed JSON-RPC message."""
        assert self._proc is not None and self._proc.stdout is not None
        # Header block.
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = await self._proc.stdout.read(1)
            if not chunk:
                raise MCPError("server closed its stdout")
            header += chunk
            if len(header) > 8192:
                raise MCPError("oversized JSON-RPC header")
        head_text = header.decode("ascii", "replace")
        length = 0
        for line in head_text.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        if length <= 0 or length > 4 * 1024 * 1024:
            raise MCPError(f"invalid Content-Length: {length}")
        body = await self._proc.stdout.readexactly(length)
        try:
            return json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise MCPError(f"bad JSON body: {exc}")


# ---- Cache of running transports keyed by server name ------------------
_transports: dict[str, StdioTransport] = {}


async def get_transport(name: str, cmd: list[str], env: dict[str, str] | None = None) -> StdioTransport:
    """Return (or create + start) a transport for the named server."""
    t = _transports.get(name)
    if t is None or not t.alive:
        t = StdioTransport(cmd=cmd, env=env, name=name)
        await t.start()
        if t.alive:
            _transports[name] = t
    return t


async def shutdown_all() -> None:
    for t in list(_transports.values()):
        await t.stop()
    _transports.clear()


__all__ = ["StdioTransport", "MCPError", "get_transport", "shutdown_all"]
