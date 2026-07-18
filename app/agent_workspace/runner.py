"""Constrained command runner (#7) — run a build/test/command confined to a
workspace, safely.

Generalizes the isolated-subprocess pattern from `app/dsa/verifier.py` to ANY
shell command, for ANY language, using runtimes installed on the host:

  - **wall-clock timeout** (always) — the process (and its group on POSIX) is
    killed on timeout.
  - **POSIX resource limits** (CPU seconds + virtual memory) via `setrlimit`
    in a pre-exec hook; Windows relies on the wall-clock timeout.
  - **workspace confinement** — `cwd` must resolve to a real directory; the
    command runs there.
  - **deny-list** — the same catastrophic-command guard the agent uses
    (`app.agent.permissions.deny_reason`) blocks `rm -rf /`, fork bombs, etc.
  - **output cap** — stdout/stderr are truncated so a runaway build log can't
    blow up memory or the model's context.

No behavior change on its own — wiring into the agent loop happens in Phase 1.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass

_DEFAULT_TIMEOUT = 120          # seconds
_DEFAULT_MAX_OUTPUT = 30_000    # chars per stream
_DEFAULT_CPU_SECONDS = 120      # POSIX RLIMIT_CPU
_DEFAULT_MEM_BYTES = 2 * 1024 * 1024 * 1024  # POSIX RLIMIT_AS (2 GB)


@dataclass
class RunResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    denied: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.denied

    def summary(self) -> str:
        if self.denied:
            return f"DENIED: {self.reason}"
        if self.timed_out:
            return f"[timed out after {self.duration_ms} ms]"
        head = f"[exit {self.exit_code} in {self.duration_ms} ms]"
        out = self.stdout
        err = (f"\n[stderr]\n{self.stderr}" if self.stderr.strip() else "")
        return f"{head}\n{out}{err}".strip()


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]"


def _bwrap_wrap(cmd: str, root: str) -> str | None:
    """Filesystem-confine an agent build/test command with bubblewrap while
    KEEPING the network (builds legitimately fetch dependencies): OS mounts
    read-only, the workspace bind-mounted writable, throwaway tmpfs /tmp and
    $HOME. Linux-with-bwrap only; None → run unconfined as before (Windows
    keeps deny-list + timeout + Job-Object limits only)."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.sandbox, "harden_runner", True)):
            return None
        from app.sandbox.executor import isolation_level
        if isolation_level() != "namespace":
            return None
    except Exception:  # noqa: BLE001 — hardening is best-effort
        return None
    import shlex
    home = os.path.expanduser("~")
    argv = [
        "bwrap", "--die-with-parent",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        # network intentionally SHARED — pip/npm installs must work
        "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/bin", "/bin",
        "--ro-bind-try", "/sbin", "/sbin",
        "--ro-bind-try", "/etc", "/etc",
        "--ro-bind-try", "/opt", "/opt",
        "--tmpfs", "/tmp",
        "--dev", "/dev",
        "--proc", "/proc",
        "--bind", root, root,
        "--tmpfs", home,
        "--setenv", "HOME", home,
        "--chdir", root,
        "/bin/sh", "-c", cmd,
    ]
    return " ".join(shlex.quote(a) for a in argv)


def _posix_limits(cpu_seconds: int, mem_bytes: int):
    """Pre-exec hook (POSIX only): cap CPU + address space, new session so we
    can kill the whole process group on timeout."""
    import resource  # type: ignore

    os.setsid()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass


async def run_in_workspace(
    command: str,
    *,
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_output: int = _DEFAULT_MAX_OUTPUT,
    env: dict | None = None,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
    mem_bytes: int = _DEFAULT_MEM_BYTES,
) -> RunResult:
    """Run `command` (shell) in `cwd`, bounded by timeout + resource limits.

    Never raises for command failure — failures are reported in the result.
    Raises only for a programming error (e.g. cwd missing).
    """
    started = time.monotonic()
    cmd = (command or "").strip()
    if not cmd:
        return RunResult(command, 0, "", "", False, 0, denied=True,
                         reason="empty command")

    # Catastrophic-command guard (reuse the agent's deny-list).
    try:
        from app.agent.permissions import deny_reason
        why = deny_reason(cmd)
    except Exception:  # noqa: BLE001 — guard import must never block
        why = None
    if why:
        return RunResult(cmd, -1, "", "", False,
                         int((time.monotonic() - started) * 1000),
                         denied=True, reason=why)

    root = os.path.realpath(cwd)
    if not os.path.isdir(root):
        return RunResult(cmd, -1, "", "", False, 0, denied=True,
                         reason=f"workspace not found: {cwd}")

    run_env = {**os.environ, **(env or {})}
    is_posix = os.name == "posix"
    # Linux hardening (2026-07-09): confine the filesystem with bwrap while
    # keeping network — untrusted uploaded-project build commands previously
    # ran with only a deny-list between them and the host filesystem.
    shell_cmd = cmd
    if is_posix:
        wrapped = _bwrap_wrap(cmd, root)
        if wrapped is not None:
            shell_cmd = wrapped
    creationflags = 0
    preexec = None
    if is_posix:
        preexec = lambda: _posix_limits(cpu_seconds, mem_bytes)  # noqa: E731
    elif sys.platform == "win32":
        # New process group so we can signal/kill the whole tree on timeout.
        creationflags = getattr(
            __import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        proc = await asyncio.create_subprocess_shell(
            shell_cmd,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=run_env,
            preexec_fn=preexec,            # POSIX only; ignored on Windows
            creationflags=creationflags,   # Windows only; 0 elsewhere
        )
    except NotImplementedError:
        # Some Windows event loops don't support subprocess on this thread.
        return _run_blocking(cmd, root, timeout, max_output, run_env, started)
    except Exception as exc:  # noqa: BLE001
        return RunResult(cmd, -1, "", f"failed to start: {exc}", False,
                         int((time.monotonic() - started) * 1000))

    timed_out = False
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        _kill(proc, is_posix)
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:  # noqa: BLE001
            out_b, err_b = b"", b""

    duration = int((time.monotonic() - started) * 1000)
    return RunResult(
        command=cmd,
        exit_code=(proc.returncode if proc.returncode is not None else -1),
        stdout=_clip(out_b.decode("utf-8", "replace"), max_output),
        stderr=_clip(err_b.decode("utf-8", "replace"), max_output),
        timed_out=timed_out,
        duration_ms=duration,
    )


def _kill(proc, is_posix: bool) -> None:
    try:
        if is_posix:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _run_blocking(cmd: str, root: str, timeout: int, max_output: int,
                  env: dict, started: float) -> RunResult:
    """Fallback for Windows event loops without async subprocess support —
    run on a (already-running) thread via blocking subprocess."""
    import subprocess
    try:
        r = subprocess.run(  # noqa: S602 — generated build commands, deny-listed
            cmd, shell=True, cwd=root, capture_output=True, text=True,
            timeout=timeout, env=env, errors="replace",
            stdin=subprocess.DEVNULL,
        )
        return RunResult(cmd, r.returncode, _clip(r.stdout or "", max_output),
                         _clip(r.stderr or "", max_output), False,
                         int((time.monotonic() - started) * 1000))
    except subprocess.TimeoutExpired:
        return RunResult(cmd, -1, "", "", True,
                         int((time.monotonic() - started) * 1000))
    except Exception as exc:  # noqa: BLE001
        return RunResult(cmd, -1, "", f"{exc}", False,
                         int((time.monotonic() - started) * 1000))


__all__ = ["RunResult", "run_in_workspace"]
