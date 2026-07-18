"""Execution sandbox (agent-orchestration R4).

`run_code(workspace_id, command)` runs generated code **only** inside the
existing `app/agent_workspace` sandbox (`run_in_workspace`) under its existing
resource/time/concurrency caps + secret redaction — nothing runs outside it
(R4.1/R4.2, Property 4). A failure yields a repair-feedback `SandboxResult`
(`verified=False`) and is never reported as verified (R4.3); when sandboxing is
disabled/unavailable the status is marked `unavailable` rather than falsely
verified (R4.4). Async + injectable runner for tests; never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class SandboxResult:
    ran: bool                 # did execution actually happen in the sandbox?
    verified: bool            # ran AND succeeded (exit 0, not timed out/denied)
    status: str               # "verified" | "failed" | "unavailable" | "disabled"
    exit_code: int = 0
    output: str = ""
    repair_feedback: str = ""  # non-empty on failure → fed back to the workflow

    @property
    def is_verified(self) -> bool:
        return self.verified


def _enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "sandbox_verify", False))
    except Exception:  # noqa: BLE001
        return False


async def run_code(workspace_id: str, command: str, *,
                   runner: Callable[..., Awaitable] | None = None,
                   force: bool = False) -> SandboxResult:
    """Execute `command` in the workspace sandbox. `runner` is injected in
    tests; in prod it defaults to `app.agent_workspace.run_in_workspace`.
    `force` bypasses the flag (used when the caller already decided to verify).
    Never raises."""
    try:
        if not force and not _enabled():
            return SandboxResult(ran=False, verified=False, status="disabled")
        cmd = (command or "").strip()
        if not cmd:
            return SandboxResult(ran=False, verified=False, status="unavailable",
                                 repair_feedback="empty command")

        if runner is None:
            try:
                from app.agent_workspace.runner import run_in_workspace
                from app.agent_workspace.materialize import workspace_path
                cwd = workspace_path(workspace_id)
                result = await run_in_workspace(cmd, cwd=cwd)
            except Exception as exc:  # noqa: BLE001 — sandbox unavailable (R4.4)
                return SandboxResult(ran=False, verified=False,
                                     status="unavailable",
                                     repair_feedback=f"sandbox unavailable: {exc}")
        else:
            result = await runner(cmd, workspace_id)

        ok = bool(getattr(result, "ok", False))
        exit_code = int(getattr(result, "exit_code", 1) or 0)
        summary = result.summary() if hasattr(result, "summary") else str(result)
        if ok:
            return SandboxResult(ran=True, verified=True, status="verified",
                                 exit_code=exit_code, output=summary)
        # Failure → repair feedback, NEVER reported as verified (R4.3).
        return SandboxResult(ran=True, verified=False, status="failed",
                             exit_code=exit_code, output=summary,
                             repair_feedback=summary)
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(ran=False, verified=False, status="unavailable",
                             repair_feedback=f"error: {exc}")


async def verify_snippet(code: str, language: str = "python") -> SandboxResult:
    """Verify a THROWAWAY script in the dedicated isolation sandbox
    (app/sandbox: bwrap namespaces on Linux → rlimits → constrained
    subprocess) — no workspace needed. Same honest contract: a failure is
    repair feedback, never falsely verified; sandbox unavailable is said so.
    """
    try:
        import asyncio
        from app.sandbox import run_code as _sbx_run
        res = await asyncio.to_thread(_sbx_run, code, language)
        if res.status == "unavailable":
            return SandboxResult(ran=False, verified=False,
                                 status="unavailable",
                                 repair_feedback=res.reason)
        out = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
        if res.ok:
            return SandboxResult(ran=True, verified=True, status="verified",
                                 exit_code=res.exit_code or 0, output=out)
        return SandboxResult(ran=True, verified=False, status="failed",
                             exit_code=res.exit_code or 1, output=out,
                             repair_feedback=out or res.reason)
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(ran=False, verified=False, status="unavailable",
                             repair_feedback=f"error: {exc}")


__all__ = ["run_code", "verify_snippet", "SandboxResult"]
