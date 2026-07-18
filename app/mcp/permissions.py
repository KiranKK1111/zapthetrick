"""Per-tool permission gating with persistent grants.

Permission model (Architecture.md):
    - Every tool has a `danger` level (low | medium | high).
    - Low-danger tools auto-grant on first use.
    - Medium prompts once, then remembers.
    - High prompts every time.
    - The user can revoke any grant from Settings -> Tools.

The store is a JSON file under `~/.zapthetrick/mcp_permissions.json`
so grants survive restarts without touching the Postgres schema.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from .registry import ToolPermission


log = logging.getLogger(__name__)


_STORE_PATH = Path.home() / ".zapthetrick" / "mcp_permissions.json"


class PermissionStore:
    """File-backed grant store. Thread-safe; tolerant of file IO errors."""

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._grants: dict[str, ToolPermission] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for tool, entry in (raw or {}).items():
                    if isinstance(entry, dict):
                        self._grants[tool] = ToolPermission(
                            granted=bool(entry.get("granted")),
                            granted_at_ms=entry.get("granted_at_ms"),
                            rationale=str(entry.get("rationale") or ""),
                        )
        except (OSError, ValueError) as exc:
            log.warning("mcp permissions: load failed (%s)", exc)
            self._grants = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        k: {
                            "granted": v.granted,
                            "granted_at_ms": v.granted_at_ms,
                            "rationale": v.rationale,
                        }
                        for k, v in self._grants.items()
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("mcp permissions: save failed (%s)", exc)

    def is_granted(self, tool: str) -> bool:
        with self._lock:
            p = self._grants.get(tool)
            return bool(p and p.granted)

    def grant(self, tool: str, rationale: str = "") -> None:
        with self._lock:
            self._grants[tool] = ToolPermission(
                granted=True,
                granted_at_ms=int(time.time() * 1000),
                rationale=rationale,
            )
            self._save()

    def revoke(self, tool: str) -> bool:
        with self._lock:
            existed = tool in self._grants
            self._grants.pop(tool, None)
            self._save()
            return existed

    def snapshot(self) -> dict[str, ToolPermission]:
        with self._lock:
            return dict(self._grants)


_default_store: PermissionStore | None = None


def default_permission_store() -> PermissionStore:
    """Lazy singleton — avoid hitting the disk at import time."""
    global _default_store
    if _default_store is None:
        _default_store = PermissionStore()
    return _default_store


__all__ = ["PermissionStore", "default_permission_store"]
