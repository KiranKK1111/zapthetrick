"""Docker-backed sandbox execution.

In docker-only mode every generated snippet is compiled + run + verified INSIDE
the `zapthetrick_sandbox` container (sandbox/Dockerfile) — one Linux image with
all 25 toolchains — instead of on the Windows host. This removes every
host-specific toolchain hack (.bat/PATHEXT, SDKROOT, JVM -Xmx, OS=Windows_NT,
scala tool_dirs, RUSTUP_HOME, GOCACHE …): inside the container they're just
clean apt/tarball installs.

Flow per run (see `run_plan`):
  1. write the source + staged files to a host temp dir,
  2. `docker exec mkdir` an ephemeral /sandbox/<id> + `docker cp` the files in,
  3. run each build/run step with `docker exec -w /sandbox/<id> … timeout <s> …`
     (the container-side `timeout` kills a runaway — killing the host-side
     `docker exec` would NOT stop the in-container process),
  4. capture stdout/stderr/exit; the first failing step is the honest result,
  5. `docker exec rm -rf /sandbox/<id>` and drop the host temp dir.

The container runs with `network_mode: none`, dropped caps, a pids cap and a
RAM-backed /sandbox (compose), so executed code is contained. Everything here
is best-effort + never raises — a missing/stopped container returns an
`unavailable` SandboxResult so verification is simply skipped with a clear reason.
"""
from __future__ import annotations

import contextvars
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid

log = logging.getLogger(__name__)

_DOCKER: str | None = None
_AVAIL: tuple[float, bool] | None = None   # (checked_at_monotonic, running)
_AVAIL_TTL = 10.0

# ── Killable-execution registry ─────────────────────────────────────────────
# A sandbox run executes in a worker thread (executor → to_thread), so cancelling
# the awaiting coroutine can't interrupt it — the `docker exec` runs to its own
# timeout. To make Stop actually stop the WORK (without touching the shared
# container), each run step is a registered, killable subprocess grouped by an
# optional run-group (the conversation id, set via the `run_group` contextvar and
# propagated into the worker thread by to_thread). `cancel_group(id)` kills every
# in-flight exec for that group + reaps its container-side process; the container
# keeps running, ready for the next task.
run_group: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sandbox_run_group", default=None)


class _Run:
    __slots__ = ("proc", "cdir", "killed")

    def __init__(self, proc: subprocess.Popen, cdir: str):
        self.proc = proc
        self.cdir = cdir
        self.killed = False


_ACTIVE: dict[str, set[_Run]] = {}
_ACTIVE_LOCK = threading.Lock()


def _register(group: str | None, entry: _Run) -> None:
    if not group:
        return
    with _ACTIVE_LOCK:
        _ACTIVE.setdefault(group, set()).add(entry)


def _unregister(group: str | None, entry: _Run) -> None:
    if not group:
        return
    with _ACTIVE_LOCK:
        s = _ACTIVE.get(group)
        if s is not None:
            s.discard(entry)
            if not s:
                _ACTIVE.pop(group, None)


def cancel_group(group: str) -> int:
    """Kill every in-flight sandbox exec for this run-group NOW. The CONTAINER is
    left running (only the exec client + its in-container process are killed), so
    the sandbox is immediately ready for the next task. Returns how many execs
    were killed. Safe to call from any thread; never raises."""
    if not group:
        return 0
    with _ACTIVE_LOCK:
        entries = list(_ACTIVE.get(group, ()))
    dbin = _docker_bin()
    cname = container_name()
    killed = 0
    dirs: set[str] = set()
    for e in entries:
        e.killed = True
        try:
            if e.proc.poll() is None:
                e.proc.kill()
                killed += 1
        except Exception:  # noqa: BLE001
            pass
        dirs.add(e.cdir)
    # The host-side kill of `docker exec` does NOT stop the process INSIDE the
    # container — reap anything still running in those ephemeral workdirs. This
    # is FIRE-AND-FORGET (Popen, never waited on): cancel_group is called from
    # the async event loop, so a blocking `subprocess.run` here would freeze the
    # whole backend for the reap's duration ("backend unreachable" on Stop).
    for cdir in dirs:
        if dbin and cdir:
            try:
                subprocess.Popen(
                    [dbin, "exec", cname, "pkill", "-9", "-f", cdir],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL)
            except Exception:  # noqa: BLE001
                pass
    if killed:
        log.info("sandbox cancel_group(%s): killed %d exec(s)", group, killed)
    return killed


def _exec_step(exec_args: list[str], timeout: float, stdin_bytes: bytes | None,
               cdir: str, group: str | None):
    """Run ONE docker-exec step as a killable, registered subprocess. Returns a
    CompletedProcess-like object, None on host-side timeout, or the sentinel
    string 'cancelled' when cancel_group killed it."""
    dbin = _docker_bin()
    if not dbin:
        return None
    try:
        proc = subprocess.Popen(
            [dbin, *exec_args],
            stdin=subprocess.PIPE if stdin_bytes is not None else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # noqa: BLE001
        log.info("docker exec spawn failed: %s", exc)
        return None
    entry = _Run(proc, cdir)
    _register(group, entry)
    try:
        out_b, err_b = proc.communicate(input=stdin_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return None
    finally:
        _unregister(group, entry)
    if entry.killed:
        return "cancelled"
    return subprocess.CompletedProcess(
        exec_args, proc.returncode, out_b or b"", err_b or b"")


def _docker_bin() -> str | None:
    global _DOCKER
    if _DOCKER is not None:
        return _DOCKER or None
    _DOCKER = shutil.which("docker") or ""
    return _DOCKER or None


def _cfg():
    try:
        from app.core.config_loader import cfg
        return cfg.sandbox
    except Exception:  # noqa: BLE001
        return None


def container_name() -> str:
    c = _cfg()
    return str(getattr(c, "container", "zapthetrick_sandbox") or
               "zapthetrick_sandbox")


def _run_docker(args: list[str], timeout: float,
                stdin: bytes | None = None) -> subprocess.CompletedProcess | None:
    """Invoke the docker CLI; None on missing binary / spawn failure / timeout."""
    dbin = _docker_bin()
    if not dbin:
        return None
    try:
        return subprocess.run(
            [dbin, *args], input=stdin, capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception as exc:  # noqa: BLE001
        log.info("docker cli failed (%s): %s", " ".join(args[:2]), exc)
        return None


def available(refresh: bool = False) -> bool:
    """True when the sandbox container is running. Cached for a few seconds so a
    burst of verifications doesn't shell out to `docker inspect` each time."""
    global _AVAIL
    now = time.monotonic()
    if not refresh and _AVAIL is not None and (now - _AVAIL[0]) < _AVAIL_TTL:
        return _AVAIL[1]
    ok = False
    cp = _run_docker(
        ["inspect", "-f", "{{.State.Running}}", container_name()], timeout=8.0)
    if cp is not None and cp.returncode == 0:
        ok = cp.stdout.decode("utf-8", "replace").strip() == "true"
    _AVAIL = (now, ok)
    return ok


def _stage_all(cname: str, cdir: str, staged: dict[str, str]) -> str | None:
    """Create the workspace and write EVERY file in ONE `docker exec` (a shell
    script piped to `sh -s`, each file base64-decoded into place). Replaces the
    old mkdir + one-`docker exec`-per-file staging (N+1 subprocess spawns → 1),
    the dominant fixed overhead per run. Returns an error string, or None on
    success."""
    import base64
    lines = ["set -e", f"mkdir -p '{cdir}'"]
    for rel, content in staged.items():
        dest = f"{cdir}/{rel}"
        parent = dest.rsplit("/", 1)[0]
        if parent != cdir:
            lines.append(f"mkdir -p '{parent}'")
        b64 = base64.b64encode((content or "").encode("utf-8")).decode("ascii")
        # b64 alphabet has no quotes → single-quoting is safe; piped via stdin so
        # there's no ARG_MAX limit on the (base64) payload.
        lines.append(f"printf %s '{b64}' | base64 -d > '{dest}'")
    cp = _run_docker(["exec", "-i", cname, "sh", "-s"], timeout=45.0,
                     stdin=("\n".join(lines) + "\n").encode("utf-8"))
    if cp is None or cp.returncode != 0:
        return (cp.stderr.decode("utf-8", "replace")[:200]
                if cp is not None else "staging failed")
    return None


def _result_from_cp(cp, cap: int, dur: int, per_step_timeout: float):
    """Map a finished `_exec_step` CompletedProcess into a SandboxResult, or a
    special value: the string 'cancelled' / 'timeout' for those terminal cases."""
    from app.sandbox.executor import SandboxResult
    if cp == "cancelled":
        return SandboxResult(status="error", backend="docker",
                             reason="stopped by user", duration_ms=dur)
    if cp is None:
        return SandboxResult(status="timeout", backend="docker",
                             reason=f"exceeded {per_step_timeout}s",
                             duration_ms=dur)
    out = cp.stdout[:cap].decode("utf-8", "replace")
    err = cp.stderr[:cap].decode("utf-8", "replace")
    rc = cp.returncode
    if rc in (124, 137):   # `timeout` SIGKILL / OOM-kill
        return SandboxResult(status="timeout", backend="docker", stdout=out,
                             stderr=err, reason=f"exceeded {per_step_timeout}s",
                             duration_ms=dur)
    return SandboxResult(status="ok" if rc == 0 else "error", exit_code=rc,
                         stdout=out, stderr=err, backend="docker",
                         duration_ms=dur)


def run_batch(main_name: str, commands: list[list[str]], code: str,
              files: dict[str, str] | None, limits,
              stdins: list[str | None]) -> list["object"]:
    """Compile the program ONCE, then run its final (RUN) step against EACH stdin
    in the SAME workspace — returns one SandboxResult per entry in `stdins`.

    This is the big speed win for stdin-style (competitive/HackerRank) verify:
    the old path called run_plan once per example, recompiling every time (a Java
    solution with 3 examples = 3 compiles + ~15 docker execs). Here the build
    steps (every command except the last) run ONCE; only the run step repeats.
    A build failure (compile error) is returned for EVERY entry so the caller
    reports it once. Never raises."""
    from app.sandbox.executor import SandboxResult  # lazy (import cycle)

    n = len(stdins)
    if not available():
        r = SandboxResult(
            status="unavailable", backend="docker",
            reason="sandbox container not running (docker compose up -d sandbox)")
        return [r] * n
    if n == 0:
        return []

    cap = int(getattr(limits, "output_kb", 256)) * 1024
    per_step_timeout = float(getattr(limits, "timeout_s", 25.0))
    cname = container_name()
    cdir = f"/sandbox/{uuid.uuid4().hex[:16]}"
    _group = run_group.get()

    try:
        staged: dict[str, str] = {main_name: code or ""}
        for rel, content in (files or {}).items():
            safe = rel.replace("\\", "/").lstrip("/")
            if ".." in safe.split("/"):
                continue
            staged[safe] = content
        err = _stage_all(cname, cdir, staged)
        if err is not None:
            r = SandboxResult(status="error", backend="docker",
                              reason=f"could not stage sources ({err})")
            return [r] * n

        # BUILD steps (all but the last command) — run ONCE. Any failure is the
        # honest result for every stdin (a compile error fails all cases).
        build_cmds, run_cmd = commands[:-1], commands[-1]
        for argv in build_cmds:
            started = time.monotonic()
            exec_args = ["exec", "-w", cdir, cname, "timeout",
                         "--signal=KILL", str(int(per_step_timeout)), *argv]
            cp = _exec_step(exec_args, per_step_timeout + 15.0, None, cdir, _group)
            dur = int((time.monotonic() - started) * 1000)
            r = _result_from_cp(cp, cap, dur, per_step_timeout)
            if r.status != "ok":
                return [r] * n

        # RUN step — once per stdin, reusing the compiled artifact.
        results: list[object] = []
        for s in stdins:
            sb = s.encode("utf-8") if s is not None else None
            started = time.monotonic()
            flag = ["-i"] if sb is not None else []
            exec_args = ["exec", *flag, "-w", cdir, cname, "timeout",
                         "--signal=KILL", str(int(per_step_timeout)), *run_cmd]
            cp = _exec_step(exec_args, per_step_timeout + 15.0, sb, cdir, _group)
            dur = int((time.monotonic() - started) * 1000)
            results.append(_result_from_cp(cp, cap, dur, per_step_timeout))
        return results
    except Exception as exc:  # noqa: BLE001 — never raise into a turn
        log.info("docker run_batch failed: %s", exc)
        r = SandboxResult(status="error", backend="docker",
                          reason=f"sandbox error ({type(exc).__name__})")
        return [r] * n
    finally:
        _run_docker(["exec", cname, "rm", "-rf", cdir], timeout=10.0)


def run_plan(main_name: str, commands: list[list[str]], code: str,
             files: dict[str, str] | None, limits,
             stdin: str | None = None) -> "object":
    """Execute a resolved (POSIX) plan inside the container. Mirrors the local
    executor's contract: run each step in order, stop at the first non-OK step,
    return a SandboxResult. `stdin` (when set) is fed to the RUN step — for
    competitive/stdin-style problems that read their input from standard in."""
    from app.sandbox.executor import SandboxResult  # lazy (avoid import cycle)

    if not available():
        return SandboxResult(
            status="unavailable", backend="docker",
            reason="sandbox container not running (docker compose up -d sandbox)")

    cap = int(getattr(limits, "output_kb", 256)) * 1024
    per_step_timeout = float(getattr(limits, "timeout_s", 25.0))
    cname = container_name()
    rid = uuid.uuid4().hex[:16]
    cdir = f"/sandbox/{rid}"

    try:
        # 1) collect the files to stage (main source + any extras).
        staged: dict[str, str] = {main_name: code or ""}
        for rel, content in (files or {}).items():
            safe = rel.replace("\\", "/").lstrip("/")
            if ".." in safe.split("/"):
                continue
            staged[safe] = content

        # 2) make the workspace, then WRITE each file from INSIDE the container
        #    (`docker exec -i … cat > dest`, content piped via stdin). `/sandbox`
        #    is a tmpfs mount, and `docker cp` cannot write into tmpfs/volume
        #    mounts ("Could not find the file …"); an in-container write does,
        #    and it sidesteps every Windows host-path quirk too.
        mk = _run_docker(["exec", cname, "mkdir", "-p", cdir], timeout=15.0)
        if mk is None or mk.returncode != 0:
            return SandboxResult(status="error", backend="docker",
                                 reason="could not create sandbox workspace")
        for rel, content in staged.items():
            dest = f"{cdir}/{rel}"
            parent = dest.rsplit("/", 1)[0]
            if parent != cdir:
                _run_docker(["exec", cname, "mkdir", "-p", parent], timeout=10.0)
            wr = _run_docker(
                ["exec", "-i", cname, "sh", "-c", f"cat > '{dest}'"],
                timeout=30.0, stdin=(content or "").encode("utf-8"))
            if wr is None or wr.returncode != 0:
                return SandboxResult(
                    status="error", backend="docker",
                    reason=f"could not stage {rel} into the sandbox")

        # 3) run each step; container-side `timeout` bounds wall-clock. `stdin`
        #    (if any) goes to the RUN step only — the LAST command; a build step
        #    (compile) never consumes it.
        res = SandboxResult(status="unavailable", reason="no command")
        _stdin_bytes = stdin.encode("utf-8") if stdin is not None else None
        _group = run_group.get()
        for _i, argv in enumerate(commands):
            _is_run = _i == len(commands) - 1
            started = time.monotonic()
            _flag = ["-i"] if (_is_run and _stdin_bytes is not None) else []
            exec_args = [
                "exec", *_flag, "-w", cdir, cname,
                "timeout", "--signal=KILL", str(int(per_step_timeout)),
                *argv,
            ]
            # Give the host-side wait a little slack over the container timeout.
            cp = _exec_step(exec_args, per_step_timeout + 15.0,
                            _stdin_bytes if _is_run else None, cdir, _group)
            dur = int((time.monotonic() - started) * 1000)
            if cp == "cancelled":
                # Stop killed this exec — the container is untouched, ready for
                # the next task.
                return SandboxResult(status="error", backend="docker",
                                     reason="stopped by user", duration_ms=dur)
            if cp is None:
                return SandboxResult(status="timeout", backend="docker",
                                     reason=f"exceeded {per_step_timeout}s",
                                     duration_ms=dur)
            out = cp.stdout[:cap].decode("utf-8", "replace")
            err = cp.stderr[:cap].decode("utf-8", "replace")
            rc = cp.returncode
            if rc == 124 or rc == 137:   # `timeout` SIGKILL / OOM-kill
                return SandboxResult(status="timeout", backend="docker",
                                     stdout=out, stderr=err,
                                     reason=f"exceeded {per_step_timeout}s",
                                     duration_ms=dur)
            res = SandboxResult(
                status="ok" if rc == 0 else "error", exit_code=rc,
                stdout=out, stderr=err, backend="docker", duration_ms=dur)
            if rc != 0:
                return res   # first failing step (e.g. compile error) is honest
        return res
    except Exception as exc:  # noqa: BLE001 — never raise into a turn
        log.info("docker run_plan failed: %s", exc)
        return SandboxResult(status="error", backend="docker",
                             reason=f"sandbox error ({type(exc).__name__})")
    finally:
        # Best-effort cleanup of the in-container workspace.
        _run_docker(["exec", cname, "rm", "-rf", cdir], timeout=10.0)
