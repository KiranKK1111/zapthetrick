"""Dedicated execution sandbox — Claude-style layered isolation
(user request #2 + SeveralFeatures.md "Sandbox Execution" scenarios).

Claude's script sandbox is: an EPHEMERAL workspace, NO network, a read-only
view of the OS, hard CPU/memory/time/output limits, a scrubbed environment,
and kill-on-timeout — run it, capture stdout/stderr/exit code, throw the
world away. This module reproduces that with the strongest isolation the
host actually offers, probed at runtime (and reported to the capability
registry):

  LEVEL "namespace"  (Linux + bubblewrap): full namespace isolation via
      `bwrap` — new user/PID/net/IPC/UTS namespaces (`--unshare-all`, so NO
      network), read-only binds of the OS, tmpfs /tmp, the workspace as the
      only writable mount, cleared env, dies with parent. This is the
      VPS/production path.
  LEVEL "rlimit"     (POSIX without bwrap): setrlimit CPU/AS/NPROC/FSIZE +
      scrubbed env + process-group kill. No mount/net isolation (honest).
  LEVEL "subprocess" (Windows/dev): scrubbed env, ephemeral workspace,
      wall-clock timeout with process-TREE kill, output caps. Resource caps
      are best-effort only — Windows is the developer convenience path; the
      deployed sandbox is Linux.

Uniform contract regardless of level: `run_code(code, language)` /
`verify_script(...)` → [SandboxResult] with status ok|error|timeout|
unavailable, captured output (capped), the backend used, and duration. The
sandbox NEVER lies about its isolation level.

Complements (does not replace) `app/agent_workspace/runner.py`, which runs
build/test commands inside a persistent materialized project workspace; this
module is the throwaway "execute/verify this script" sandbox.
"""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Language → (main filename, argv builder). `{exe}` is resolved per-host.
_LANGUAGES: dict[str, tuple[str, list[str]]] = {
    "python": ("main.py", [sys.executable or "python3", "-I", "-B", "main.py"]),
    "javascript": ("main.js", ["node", "main.js"]),
    "node": ("main.js", ["node", "main.js"]),
    "bash": ("main.sh", ["bash", "main.sh"]),
    "sh": ("main.sh", ["sh", "main.sh"]),
    "powershell": ("main.ps1", ["powershell", "-NoProfile", "-NonInteractive",
                                "-ExecutionPolicy", "Bypass", "-File",
                                "main.ps1"]),
    "ruby": ("main.rb", ["ruby", "main.rb"]),
    "php": ("main.php", ["php", "main.php"]),
}

import re as _re

# Compiled languages need a compile step THEN a run step (cross-platform argv).
# Java is the interview default; Go's `go run` is a single cross-platform cmd.
_COMPILED_LANGS = ("java", "go")


def _java_class(code: str) -> str:
    """The public class name (so the file can be named <Class>.java and run as
    `java <Class>`), else 'Main'."""
    m = _re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_]\w*)",
                   code or "")
    if m:
        return m.group(1)
    m = _re.search(r"\bclass\s+([A-Za-z_]\w*)", code or "")
    return m.group(1) if m else "Main"


def _lang_plan(lang: str, code: str) -> tuple[str, list[list[str]]] | None:
    """(main filename, [argv, ...]) for `lang`. Interpreted → one run command;
    compiled → [compile, run]. None when the language isn't supported.

    Delegates to the data-driven registry (app/sandbox/lang_registry.py) which
    covers ~35 languages; the legacy `_LANGUAGES` table is kept as a fallback for
    a couple of shell dialects the registry folds together."""
    try:
        from app.sandbox import lang_registry
        p = lang_registry.plan(lang, code)
        if p is not None:
            return p
    except Exception:  # noqa: BLE001 — never let a registry hiccup break a run
        pass
    if lang in _LANGUAGES:
        name, argv = _LANGUAGES[lang]
        return name, [argv]
    return None


_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "COMSPEC",
                  # Safe, non-sensitive Windows system vars some toolchains
                  # (notably `dotnet run`) need for path resolution — omitting
                  # them NPEs with "path1 null".
                  "SYSTEMDRIVE", "PROGRAMFILES", "PROGRAMFILES(X86)",
                  "PROGRAMDATA", "WINDIR", "PROCESSOR_ARCHITECTURE",
                  "NUMBER_OF_PROCESSORS",
                  # `OS` (=Windows_NT) gates the modern arg-parsing branch in
                  # legacy tool launchers: Groovy's startGroovy.bat falls into a
                  # Win9x code path that HANGS forever when %OS% is unset.
                  "OS",
                  "PATHEXT", "TEMP", "TMP")


@dataclass
class SandboxLimits:
    timeout_s: float = 10.0
    cpu_s: int = 8                 # POSIX rlimit only
    memory_mb: int = 512           # POSIX rlimit only
    output_kb: int = 256           # stdout+stderr cap (all levels)
    max_files_mb: int = 32         # FSIZE rlimit (POSIX)

    @classmethod
    def from_config(cls) -> "SandboxLimits":
        try:
            from app.core.config_loader import cfg
            s = cfg.sandbox
            return cls(timeout_s=float(s.timeout_s), cpu_s=int(s.cpu_s),
                       memory_mb=int(s.memory_mb),
                       output_kb=int(s.output_kb),
                       max_files_mb=int(s.max_files_mb))
        except Exception:  # noqa: BLE001
            return cls()


@dataclass
class SandboxResult:
    status: str                    # ok | error | timeout | unavailable
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    backend: str = ""              # namespace | rlimit | subprocess
    duration_ms: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {"status": self.status, "exit_code": self.exit_code,
                "stdout": self.stdout, "stderr": self.stderr,
                "backend": self.backend, "duration_ms": self.duration_ms,
                "reason": self.reason}


# ---- isolation-level probe --------------------------------------------------
_level_cache: str | None = None


def _bwrap_path() -> str | None:
    return shutil.which("bwrap")


def _bwrap_works() -> bool:
    """FUNCTIONAL probe: bwrap being installed isn't enough — inside Docker
    the runtime's seccomp policy may block user-namespace creation. Run a
    trivial sandboxed `true` once; only a successful run earns the
    'namespace' level (zero-touch deployments self-detect the real level)."""
    bw = _bwrap_path()
    if not bw:
        return False
    try:
        r = subprocess.run(
            [bw, "--die-with-parent", "--unshare-all",
             "--ro-bind", "/", "/", "--", "/bin/true"],
            capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def isolation_level(refresh: bool = False) -> str:
    """Strongest isolation ACTUALLY WORKING on this host (probed once,
    cached). Never reports namespace isolation it didn't demonstrate."""
    global _level_cache
    if _level_cache is not None and not refresh:
        return _level_cache
    if os.name == "posix" and _bwrap_works():
        _level_cache = "namespace"
    elif os.name == "posix":
        try:
            import resource  # noqa: F401 — POSIX-only stdlib
            _level_cache = "rlimit"
        except Exception:  # noqa: BLE001
            _level_cache = "subprocess"
    else:
        _level_cache = "subprocess"
    return _level_cache


# ---- backend implementations -------------------------------------------------
def build_bwrap_argv(workdir: str, argv: list[str], *,
                     share_net: bool = False) -> list[str]:
    """The bubblewrap invocation (pure function — unit-testable anywhere).

    Read-only OS view, tmpfs /tmp, the workspace as the only writable mount,
    ALL namespaces unshared (=> no network), cleared environment, new session,
    dies with the parent — the Claude-sandbox shape. `share_net=True` keeps
    the network (dependency-install verify tier only)."""
    bw = [_bwrap_path() or "bwrap",
          "--die-with-parent", "--new-session", "--unshare-all",
          *(["--share-net"] if share_net else []),
          "--clearenv",
          "--setenv", "PATH", "/usr/bin:/bin:/usr/local/bin",
          "--setenv", "HOME", "/work",
          "--setenv", "LANG", "C.UTF-8",
          "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
          "--bind", workdir, "/work", "--chdir", "/work"]
    for ro in ("/usr", "/bin", "/lib", "/lib64", "/etc/ssl", "/etc/resolv.conf"):
        if pathlib.Path(ro).exists():
            bw += ["--ro-bind", ro, ro]
    return bw + ["--"] + argv


def _posix_preexec(limits: SandboxLimits):
    """setrlimit + own process group (POSIX child hook)."""
    def _hook() -> None:
        import resource
        os.setsid()
        mem = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_s, limits.cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        fsz = limits.max_files_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsz, fsz))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:  # noqa: BLE001 — not on all platforms
            pass
    return _hook


_SWIFT_SDK: str | None = None
_SWIFT_SDK_DONE = False


def _swift_sdkroot() -> str | None:
    """Locate the Swift Windows platform SDK once (…/Platforms/<ver>/
    Windows.platform/Developer/SDKs/Windows.sdk). Cached; returns None off
    Windows or when Swift isn't installed."""
    global _SWIFT_SDK, _SWIFT_SDK_DONE
    if _SWIFT_SDK_DONE:
        return _SWIFT_SDK
    _SWIFT_SDK_DONE = True
    if os.name == "nt":
        import glob
        roots = [os.path.join(os.environ.get("LOCALAPPDATA", ""),
                              "Programs", "Swift", "Platforms"),
                 r"C:\Library\Developer\Platforms"]
        for root in roots:
            hits = sorted(glob.glob(os.path.join(
                root, "*", "Windows.platform", "Developer",
                "SDKs", "Windows.sdk")), reverse=True)
            if hits:
                _SWIFT_SDK = hits[0]
                break
    return _SWIFT_SDK


def _scrubbed_env(workdir: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() in _SAFE_ENV_KEYS}
    # Resolve runtimes against tool_dirs + the LIVE persisted PATH (registry on
    # Windows), not just this process's launch-time snapshot — so a toolchain
    # installed after the server started still runs, and execution resolves the
    # exact binary the availability check (lang_registry) reported. Single source
    # of truth in lang_registry.search_path().
    from app.sandbox import lang_registry as _lr
    env["PATH"] = _lr.search_path()
    env.pop("Path", None)  # normalize to one PATH key
    env["HOME"] = workdir
    env["PYTHONIOENCODING"] = "utf-8"
    # Toolchains that need a writable cache/home but whose default dir was
    # scrubbed away (Go can't find GOCACHE; .NET/NuGet need a home). GOCACHE is
    # PERSISTENT (a fixed temp dir, NOT the ephemeral workspace) so Go compiles
    # the std lib once and reuses it — a per-run cache recompiles everything and
    # blows the timeout. Harmless for languages that ignore these.
    _gocache = os.path.join(tempfile.gettempdir(), "dtt-sbx-gocache")
    try:
        os.makedirs(_gocache, exist_ok=True)
    except Exception:  # noqa: BLE001
        _gocache = os.path.join(workdir, ".gocache")
    env["GOCACHE"] = _gocache
    env["GOPATH"] = os.path.join(workdir, ".gopath")
    env["GOTOOLCHAIN"] = "local"   # never try to fetch a different Go version
    env["GOFLAGS"] = "-mod=mod"
    env["GOPROXY"] = "off"         # a self-contained hello never needs a fetch
    # Swift on Windows can't locate its own stdlib unless SDKROOT points at the
    # installed Windows platform SDK ("unable to load standard library for target
    # x86_64-unknown-windows-msvc" otherwise) — the scrubbed env drops the value
    # the installer set. Discover it once and reuse. Harmless for other langs.
    _sdk = _swift_sdkroot()
    if _sdk:
        env["SDKROOT"] = _sdk
    _dotnethome = os.path.join(tempfile.gettempdir(), "dtt-sbx-dotnet")
    try:
        os.makedirs(_dotnethome, exist_ok=True)
    except Exception:  # noqa: BLE001
        _dotnethome = workdir
    env["DOTNET_CLI_HOME"] = _dotnethome  # PERSISTENT (first-run state reused)
    # .NET path resolution NPEs without these — point them at the sandbox's own
    # dotnet home (not the real user profile) so isolation is preserved.
    env.setdefault("USERPROFILE", _dotnethome)
    env.setdefault("APPDATA", os.path.join(_dotnethome, "AppData", "Roaming"))
    env.setdefault("LOCALAPPDATA", os.path.join(_dotnethome, "AppData", "Local"))
    env["NUGET_PACKAGES"] = os.path.join(_dotnethome, "nuget")
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    env["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1"
    env["DOTNET_NOLOGO"] = "1"
    env["DOTNET_ADD_GLOBAL_TOOLS_TO_PATH"] = "0"
    env["DOTNET_SKIP_WORKLOAD_INTEGRITY_CHECK"] = "1"
    # JVM tool front-ends (kotlinc, scala, groovy) default to a max heap of ¼ of
    # physical RAM and try to RESERVE it up front; under the per-process Job
    # Object memory cap that commit fails ("paging file too small" / the JVM
    # never starts). Bound every JVM launch to a modest heap so it fits. Applies
    # to all JVM tools via the standard env hook; ignored by non-JVM langs.
    env["JAVA_TOOL_OPTIONS"] = "-Xmx512m -Xss16m"
    # rustc/cargo on Windows are rustup proxy shims that read the default
    # toolchain from %RUSTUP_HOME%\settings.toml (default ~/.rustup). We rewrite
    # USERPROFILE for .NET isolation above, which would send rustup looking in
    # the wrong home ("could not choose a version of rustc… no default"). Pin
    # RUSTUP_HOME/CARGO_HOME to the REAL user home so the toolchain resolves.
    _real_home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    if _real_home:
        env.setdefault("RUSTUP_HOME", os.path.join(_real_home, ".rustup"))
        env.setdefault("CARGO_HOME", os.path.join(_real_home, ".cargo"))
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


_UNSHARE_NET: bool | None = None


def _unshare_net_available() -> bool:
    """Cached probe: can this host isolate the network namespace with plain
    `unshare -rn` (used at the rlimit level, where bwrap is absent)? Windows
    has no equivalent — the subprocess level stays network-open there (Job
    Objects cap memory/CPU; the deny-list and timeout bound behavior)."""
    global _UNSHARE_NET
    if _UNSHARE_NET is not None:
        return _UNSHARE_NET
    if os.name != "posix" or shutil.which("unshare") is None:
        _UNSHARE_NET = False
        return False
    try:
        r = subprocess.run(["unshare", "-rn", "true"],
                           capture_output=True, timeout=5)
        _UNSHARE_NET = r.returncode == 0
    except Exception:  # noqa: BLE001
        _UNSHARE_NET = False
    return _UNSHARE_NET


def _win_job_limits(proc, limits: "SandboxLimits"):
    """Windows enforcement of memory/CPU via a Job Object (2026-07-09) —
    mirrors the POSIX rlimits, closing the "wall-clock timeout only" gap at
    the subprocess isolation level. Best-effort: returns the job handle (keep
    it referenced) or None when pywin32 is absent / assignment fails."""
    try:
        import win32api
        import win32job
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)
        basic = info["BasicLimitInformation"]
        basic["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_PROCESS_TIME
            | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        # 100-nanosecond units.
        basic["PerProcessUserTimeLimit"] = int(limits.cpu_s * 10_000_000)
        basic["ActiveProcessLimit"] = 32
        info["ProcessMemoryLimit"] = int(limits.memory_mb) * 1024 * 1024
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info)
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        handle = win32api.OpenProcess(0x0100 | 0x0001, False, proc.pid)
        win32job.AssignProcessToJobObject(job, handle)
        return job
    except Exception:  # noqa: BLE001 — enforcement is best-effort on Windows
        return None


def run_command(argv: list[str], *, workdir: str,
                limits: SandboxLimits | None = None,
                allow_network: bool = False,
                extra_env: dict | None = None,
                stdin_data: str | None = None) -> SandboxResult:
    """Run one command inside the sandbox at the strongest available level.
    `allow_network=True` keeps the network reachable (dependency installs);
    `extra_env` adds variables on top of the scrubbed environment. `stdin_data`
    is fed to the process's standard input (stdin-style problems)."""
    limits = limits or SandboxLimits.from_config()
    level = isolation_level()
    cap = limits.output_kb * 1024
    started = time.monotonic()

    popen_kw: dict = {
        "cwd": workdir,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": {**_scrubbed_env(workdir), **(extra_env or {})},
    }
    if stdin_data is not None:
        popen_kw["stdin"] = subprocess.PIPE
    cmd = list(argv)
    # A COMPILED binary lives in the workspace, not on PATH — Windows subprocess
    # doesn't search the cwd (and `./prog` is POSIX-only), so resolve a
    # workspace-local first arg to its absolute path so it can actually run.
    if level != "namespace" and cmd:
        _first = cmd[0].replace("\\", "/").lstrip("./")
        _resolved = False
        for _cand in (os.path.join(workdir, _first),
                      os.path.join(workdir, _first + ".exe")):
            if os.path.isfile(_cand):
                cmd[0] = _cand
                _resolved = True
                break
        # On Windows, subprocess.Popen resolves a bare name only against .exe —
        # it ignores PATHEXT, so npm/JVM launchers shipped as .bat/.cmd (tsc,
        # kotlinc, scala, elixir, groovy, dart) raise WinError 2. shutil.which
        # honors PATHEXT; resolve the launcher on the sandbox's own PATH so those
        # runtimes are actually found. (POSIX: which() is a harmless no-op here.)
        if (not _resolved and os.name == "nt"
                and not os.path.isabs(cmd[0]) and os.sep not in cmd[0]):
            _env = popen_kw["env"]
            _path = _env.get("Path") or _env.get("PATH") or os.environ.get("PATH", "")
            _hit = shutil.which(cmd[0], path=_path)
            if _hit:
                cmd[0] = _hit
    if level == "namespace":
        cmd = build_bwrap_argv(workdir, argv, share_net=allow_network)
        popen_kw["env"] = {}                     # bwrap --clearenv owns env
        if extra_env:
            # bwrap clears env; forward extras via --setenv.
            _extras: list[str] = []
            for k, v in extra_env.items():
                _extras += ["--setenv", str(k), str(v)]
            cmd = cmd[:1] + _extras + cmd[1:]
    elif level == "rlimit":
        popen_kw["preexec_fn"] = _posix_preexec(limits)
        # No bwrap on this POSIX host — still cut the network off when the
        # kernel allows an unprivileged netns (2026-07-09).
        if not allow_network and _unshare_net_available():
            cmd = ["unshare", "-rn", *cmd]
    else:  # subprocess (Windows)
        if os.name == "nt":
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(cmd, **popen_kw)
    except FileNotFoundError as exc:
        return SandboxResult(status="unavailable", backend=level,
                             reason=f"runtime not found: {exc}")
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(status="error", backend=level,
                             reason=str(exc)[:200])
    # Windows subprocess level: enforce memory/CPU via a Job Object (kept
    # referenced until the process finishes; kill-on-close reaps strays).
    _job = (_win_job_limits(proc, limits)
            if level == "subprocess" and os.name == "nt" else None)
    try:
        out, err = proc.communicate(
            input=stdin_data.encode("utf-8") if stdin_data is not None else None,
            timeout=limits.timeout_s)
        status = "ok" if proc.returncode == 0 else "error"
        return SandboxResult(
            status=status, exit_code=proc.returncode,
            stdout=out[:cap].decode("utf-8", errors="replace"),
            stderr=err[:cap].decode("utf-8", errors="replace"),
            backend=level,
            duration_ms=int((time.monotonic() - started) * 1000))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            out = err = b""
        return SandboxResult(
            status="timeout", exit_code=None,
            stdout=(out or b"")[:cap].decode("utf-8", errors="replace"),
            stderr=(err or b"")[:cap].decode("utf-8", errors="replace"),
            backend=level, reason=f"exceeded {limits.timeout_s}s",
            duration_ms=int((time.monotonic() - started) * 1000))


def run_code(code: str, language: str = "python", *,
             files: dict[str, str] | None = None,
             limits: SandboxLimits | None = None,
             version: str | None = None,
             stdin: str | None = None) -> SandboxResult:
    """Execute a script in an EPHEMERAL sandbox workspace and throw the
    workspace away. `files` stages extra inputs next to the script (small
    multi-file projects). `version` pins a toolchain major version in the docker
    backend (e.g. Python "2.7"); None → the container default. `stdin` feeds the
    program's standard input (competitive/stdin-style problems)."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.sandbox, "enabled", True)):
            return SandboxResult(status="unavailable",
                                 reason="sandbox disabled by config")
        allowed = list(getattr(cfg.sandbox, "languages", []) or [])
        backend = str(getattr(cfg.sandbox, "backend", "local") or "local").lower()
    except Exception:  # noqa: BLE001
        allowed = []
        backend = "local"
    lang = (language or "python").strip().lower()
    limits = limits or SandboxLimits.from_config()

    # DOCKER backend: compile + run everything inside the Linux sandbox container
    # (one image with all 25 toolchains) — no host toolchains, clean isolation.
    if backend == "docker":
        from app.sandbox import docker_exec, lang_registry
        if allowed and lang not in allowed:
            return SandboxResult(status="unavailable",
                                 reason=f"language not supported: {language}")
        if not lang_registry.container_supports(lang):
            return SandboxResult(
                status="unavailable",
                reason=f"language not runnable in the sandbox: {language}")
        dplan = lang_registry.plan(lang, code or "", posix=True, version=version)
        if dplan is None:
            return SandboxResult(status="unavailable",
                                 reason=f"language not supported: {language}")
        _dmain, _dcmds = dplan
        # Small multi-file projects: fold extra staged source files into the
        # compile (gcc/javac/go run …).
        _dcmds = lang_registry.augment_multifile(lang, _dmain, _dcmds, files)
        # The container runs with network_mode:none, so the host-side hardening
        # pre-scan is unnecessary here — the OS boundary already contains it.
        return docker_exec.run_plan(_dmain, _dcmds, code or "", files, limits,
                                    stdin=stdin)

    plan = _lang_plan(lang, code or "")
    if plan is None or (allowed and lang not in allowed):
        return SandboxResult(status="unavailable",
                             reason=f"language not supported: {language}")
    main_name, commands = plan
    # Sandbox hardening (P4 #19): on a backend that can't isolate the network
    # (rlimit / subprocess), refuse a script that opens a socket, shells out to a
    # net tool, or reads system secrets BEFORE it runs — the OS wouldn't contain
    # it. Namespace (bwrap) already denies those, so there it's advisory only.
    try:
        from app.core.config_loader import cfg as _cfg
        if bool(getattr(_cfg.sandbox, "hardening_enabled", True)):
            from app.sandbox import hardening
            rep = hardening.assess(
                code or "", net_isolated=isolation_level() == "namespace")
            if rep.blocked:
                top = rep.findings[0] if rep.findings else None
                return SandboxResult(
                    status="error",
                    backend=isolation_level(),
                    reason=("blocked by sandbox policy: "
                            f"{top.detail if top else 'escape attempt'} — "
                            "network/secret access is not permitted on this "
                            "sandbox backend"),
                )
    except Exception:  # noqa: BLE001 — hardening must never break a clean run
        pass
    ws = tempfile.mkdtemp(prefix="dtt-sbx-")
    try:
        pathlib.Path(ws, main_name).write_text(code or "", encoding="utf-8")
        for rel, content in (files or {}).items():
            p = pathlib.Path(ws) / pathlib.PurePosixPath(rel)
            # Confine staged files to the workspace (no ../ escape).
            if ".." in p.relative_to(ws).parts:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        # Compiled languages: run each step (compile → run) in sequence; the
        # first non-OK step (e.g. a compile error) is the honest result — a
        # compile failure is "not verified" with the compiler output as
        # repair feedback, never falsely reported as a successful run.
        res = SandboxResult(status="unavailable", reason="no command")
        for _i, argv in enumerate(commands):
            _run_stdin = stdin if _i == len(commands) - 1 else None
            res = run_command(argv, workdir=ws, limits=limits,
                              stdin_data=_run_stdin)
            if not res.ok:
                return res
        return res
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def run_batch(code: str, language: str = "python", *,
              files: dict[str, str] | None = None,
              limits: SandboxLimits | None = None,
              version: str | None = None,
              stdins: list[str | None] | None = None) -> list[SandboxResult]:
    """Run `code` against MANY stdin inputs, compiling ONCE — returns one
    SandboxResult per entry in `stdins` (aligned by index).

    On the docker backend this compiles + stages a single workspace and reuses
    the artifact for every input (the compile-once-run-many win for stdin-style
    verification). On the local backend it falls back to a per-input run_code
    loop (correct, just not compile-shared — local isn't the hot path)."""
    stdins = list(stdins or [])
    if not stdins:
        return []
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.sandbox, "enabled", True)):
            return [SandboxResult(status="unavailable",
                                  reason="sandbox disabled by config")] * len(stdins)
        allowed = list(getattr(cfg.sandbox, "languages", []) or [])
        backend = str(getattr(cfg.sandbox, "backend", "local") or "local").lower()
    except Exception:  # noqa: BLE001
        allowed, backend = [], "local"
    lang = (language or "python").strip().lower()
    limits = limits or SandboxLimits.from_config()

    if backend == "docker":
        from app.sandbox import docker_exec, lang_registry
        if allowed and lang not in allowed:
            return [SandboxResult(status="unavailable",
                                  reason=f"language not supported: {language}")] * len(stdins)
        if not lang_registry.container_supports(lang):
            return [SandboxResult(
                status="unavailable",
                reason=f"language not runnable in the sandbox: {language}")] * len(stdins)
        dplan = lang_registry.plan(lang, code or "", posix=True, version=version)
        if dplan is None:
            return [SandboxResult(status="unavailable",
                                  reason=f"language not supported: {language}")] * len(stdins)
        _dmain, _dcmds = dplan
        _dcmds = lang_registry.augment_multifile(lang, _dmain, _dcmds, files)
        return docker_exec.run_batch(_dmain, _dcmds, code or "", files, limits,
                                     stdins)
    # Local backend: correct but not compile-shared.
    return [run_code(code, language, files=files, limits=limits,
                     version=version, stdin=s) for s in stdins]


def verify_script(code: str, language: str = "python", *,
                  expected_stdout: str | None = None,
                  limits: SandboxLimits | None = None) -> SandboxResult:
    """The doc's verification loop primitive: EXECUTE the generated script and
    judge the run — nonzero exit / crash / timeout = not verified; when
    `expected_stdout` is given, the trimmed output must match too."""
    res = run_code(code, language, limits=limits)
    if res.ok and expected_stdout is not None \
            and res.stdout.strip() != expected_stdout.strip():
        res.status = "error"
        res.reason = (f"output mismatch: expected "
                      f"{expected_stdout.strip()!r}, got {res.stdout.strip()!r}")
    return res


__all__ = ["SandboxLimits", "SandboxResult", "run_code", "run_command",
           "verify_script", "isolation_level", "build_bwrap_argv"]
