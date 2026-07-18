"""Workspace CRUD — file-backed.

A `workspaces.json` file under `~/.zapthetrick/` holds every saved
workspace and an `active` pointer. The repo is fully synchronous —
no DB lookup, no network — so the Settings screen can rely on it
even before the active workspace's DB is reachable.

Schema:
    {
      "active": "default",
      "workspaces": {
        "default": {
          "name": "default",
          "relational": {"driver": "postgres", "host": "localhost", ...},
          "vector":     {"driver": "pgvector", "url": "..."},
          "cache":      {"driver": "redis",    "url": "..."},
          "blob":       {"driver": "filesystem", "path": "..."}
        }
      }
    }

Secrets stay in this file; on Windows / macOS we recommend the user
move it onto an encrypted volume. A future task can shift secrets
to the OS keyring without changing this API.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path


log = logging.getLogger(__name__)


_STORE_PATH = Path.home() / ".zapthetrick" / "workspaces.json"


@dataclass
class Workspace:
    name: str
    relational: dict = field(default_factory=dict)
    vector: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    blob: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "relational": self.relational,
            "vector": self.vector,
            "cache": self.cache,
            "blob": self.blob,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Workspace":
        return cls(
            name=str(raw.get("name") or "default"),
            relational=raw.get("relational") or {},
            vector=raw.get("vector") or {},
            cache=raw.get("cache") or {},
            blob=raw.get("blob") or {},
        )


class WorkspaceRepo:
    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._data: dict = {"active": "default", "workspaces": {}}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {
                        "active": raw.get("active") or "default",
                        "workspaces": raw.get("workspaces") or {},
                    }
        except (OSError, ValueError) as exc:
            log.warning("workspace repo: load failed (%s)", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("workspace repo: save failed (%s)", exc)

    def list(self) -> list[Workspace]:
        with self._lock:
            return [
                Workspace.from_dict(w)
                for w in (self._data.get("workspaces") or {}).values()
            ]

    def get(self, name: str) -> Workspace | None:
        with self._lock:
            raw = (self._data.get("workspaces") or {}).get(name)
            return Workspace.from_dict(raw) if raw else None

    def active(self) -> Workspace | None:
        with self._lock:
            return self.get(self._data.get("active") or "default")

    def upsert(self, ws: Workspace) -> None:
        with self._lock:
            self._data.setdefault("workspaces", {})[ws.name] = ws.to_dict()
            self._save()

    def set_active(self, name: str) -> bool:
        with self._lock:
            if name not in (self._data.get("workspaces") or {}):
                return False
            self._data["active"] = name
            self._save()
            return True

    def delete(self, name: str) -> bool:
        with self._lock:
            removed = (self._data.get("workspaces") or {}).pop(name, None) is not None
            if removed and self._data.get("active") == name:
                self._data["active"] = next(
                    iter(self._data.get("workspaces") or {}),
                    "default",
                )
            self._save()
            return removed


_default: WorkspaceRepo | None = None


def default_workspace_repo() -> WorkspaceRepo:
    global _default
    if _default is None:
        _default = WorkspaceRepo()
    return _default


__all__ = ["Workspace", "WorkspaceRepo", "default_workspace_repo"]
