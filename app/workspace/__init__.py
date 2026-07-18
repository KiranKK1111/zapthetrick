"""Workspace abstraction — Architecture.md §"Workspace/database config".

A workspace is a *named bundle* of (database + vector store + cache +
blob store) credentials. The UI lets the user maintain several
workspaces and switch between them; the backend resolves the active
workspace and re-routes every storage call through that workspace's
drivers.

This subpackage is the abstraction layer the existing single-DB
code can opt into incrementally. The repo today still reads
`cfg.database.postgres` directly; once the UI is in place, the
settings route can switch to writing the active workspace and the
storage code will read from the workspace's drivers.

Modules:
    drivers   — declarative driver registry for the supported stacks
                (postgres, mysql, sqlite, lancedb).
    repo      — CRUD over a `workspaces.json` file under ~/.zapthetrick/
    test      — 10-step connection probe per the doc's "connection test".
"""
from .drivers import DRIVERS, Driver, DriverKind
from .repo import Workspace, WorkspaceRepo, default_workspace_repo
from .test import probe_workspace


__all__ = [
    "DRIVERS",
    "Driver",
    "DriverKind",
    "Workspace",
    "WorkspaceRepo",
    "default_workspace_repo",
    "probe_workspace",
]
