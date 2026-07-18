"""Workspace grouping (workspace-and-artifacts R1/R2).

NOTE: named `wsgroup` (not `workspace`) to avoid colliding with the existing
`app.workspace` package, which is the DATABASE-connection-profile feature
(DRIVERS / Workspace / default_workspace_repo). This package is the
product-level Workspace that groups conversations + files + artifacts.

A Workspace is an ADDITIVE owner over the existing `Session` rows — ownership is
recorded in the existing `session_metadata` JSONB under `workspace_id` (no schema
migration), and a transparent Default_Workspace wraps a user's pre-existing flat
conversations so no-workspace behavior is byte-for-byte today's (Property 1/3).
"""
from .manager import (
    DEFAULT_WORKSPACE_ID, WorkspaceManager, session_workspace_id, belongs_to,
)

__all__ = [
    "DEFAULT_WORKSPACE_ID", "WorkspaceManager", "session_workspace_id",
    "belongs_to",
]
