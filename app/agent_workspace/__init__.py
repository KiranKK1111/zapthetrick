"""Agent workspace primitives — the backend pieces that let the chat agent
operate on a real project folder (build-from-doc, fix/optimize an uploaded
codebase).

Phase 0 ships two safety/capability primitives with NO behavior change:
  - `buildsys`  — detect the project's build/test/lint/run commands.
  - `runner`    — run a command confined to a workspace, with a wall-clock
                  timeout, POSIX resource limits, output caps, and the agent
                  deny-list applied.

Later phases add: verification loop (Phase 1), archive→workspace materializer
(Phase 2), the chat agent-run endpoint (Phase 3), and the rest of §9.
"""
from __future__ import annotations

from .buildsys import BuildSystem, detect_build_system, detect_build_systems
from .brain import (
    append_scratchpad,
    brain_context,
    clear_scratchpad,
    ensure_brain,
    preferred_model,
    read_brain,
    read_scratchpad,
    record_decision,
    remember_run,
    scratchpad_context,
)
from .materialize import (
    MaterializeResult,
    cleanup,
    diff_summary,
    enforce_quota,
    fresh_workspace,
    git_init_baseline,
    materialize_archive,
    package_workspace,
    semantic_change_summary,
    workspace_exists,
    workspace_path,
)
from .redact import redact_event, redact_secrets
from .runner import RunResult, run_in_workspace
from .verify import VerifyReport, VerifyStep, verify_workspace
from .gitflow import GitResult, run_git_workflow

__all__ = [
    "BuildSystem",
    "detect_build_system",
    "detect_build_systems",
    "RunResult",
    "run_in_workspace",
    "VerifyReport",
    "VerifyStep",
    "verify_workspace",
    "MaterializeResult",
    "materialize_archive",
    "workspace_path",
    "workspace_exists",
    "fresh_workspace",
    "package_workspace",
    "git_init_baseline",
    "diff_summary",
    "semantic_change_summary",
    "cleanup",
    "enforce_quota",
    "redact_secrets",
    "redact_event",
    "brain_context",
    "ensure_brain",
    "read_brain",
    "record_decision",
    "remember_run",
    "preferred_model",
    "append_scratchpad",
    "read_scratchpad",
    "scratchpad_context",
    "clear_scratchpad",
    "GitResult",
    "run_git_workflow",
]
