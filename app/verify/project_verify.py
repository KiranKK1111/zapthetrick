"""Project verification in the dedicated sandbox (user ask: "everything end
to end must be planned, built, VERIFIED and TESTED in sandbox — like Claude").

Given the files of a generated project (parsed from the chat build's archive
or staged from a workspace), this module actually EXECUTES checks inside
`app/sandbox` (namespace-isolated on Linux):

  1. syntax  — every .py py_compile'd; every .json parsed; every .yml/.yaml
               parsed; every .js `node --check`ed (when node exists);
  2. tests   — when the project ships tests and the stack supports it,
               `pytest -q` runs in the sandbox (bounded, no network).

The verdict is honest three-state: verified | failed (with repair feedback —
compiler/test output) | partial (some checks impossible on this host, e.g. no
JVM: what CAN be checked was; the rest is listed as skipped). It never
reports "verified" for anything it didn't actually run.
"""
from __future__ import annotations

import io
import json
import logging
import pathlib
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_MAX_FILES = 400
_MAX_FILE_BYTES = 512 * 1024

# The in-sandbox driver: compiles/parses everything it can and prints ONE
# JSON verdict line. Runs with the sandbox's isolation + limits.
_DRIVER = r"""
import json, os, py_compile, shutil, subprocess, sys
failures, checked, skipped = [], 0, []
_node = shutil.which("node")
for root, _dirs, names in os.walk("."):
    if any(p in root for p in (".git", "__pycache__", "node_modules")):
        continue
    for n in names:
        p = os.path.join(root, n)
        rel = os.path.relpath(p, ".")
        if rel == "_dtt_verify_driver.py":
            continue
        try:
            if n.endswith(".py"):
                checked += 1
                py_compile.compile(p, doraise=True)
            elif n.endswith(".json"):
                checked += 1
                json.load(open(p, encoding="utf-8"))
            elif n.endswith((".yml", ".yaml")):
                try:
                    import yaml
                except Exception:
                    skipped.append(rel + " (no yaml parser)")
                    continue
                checked += 1
                yaml.safe_load(open(p, encoding="utf-8"))
            elif n.endswith((".js", ".mjs", ".cjs")):
                if not _node:
                    skipped.append(rel + " (no node)")
                    continue
                checked += 1
                r = subprocess.run([_node, "--check", p],
                                   capture_output=True, text=True,
                                   timeout=15)
                if r.returncode != 0:
                    failures.append({"file": rel,
                                     "error": (r.stderr or r.stdout)[:400]})
        except Exception as exc:
            failures.append({"file": rel, "error": str(exc)[:400]})
print(json.dumps({"checked": checked, "failures": failures,
                  "skipped": skipped}))
"""

# Entrypoint candidates for the smoke run, in preference order.
_ENTRYPOINTS = ("main.py", "app.py", "run.py", "server.py",
                "index.js", "server.js", "app.js")


@dataclass
class ProjectVerification:
    status: str = "skipped"           # verified | failed | partial | skipped
    checked: int = 0
    failures: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    tests: str = "none"               # passed | failed | none | skipped
    test_output: str = ""
    backend: str = ""
    stack: dict | None = None
    # Entrypoint smoke run: the program was actually EXECUTED in the sandbox.
    # passed | long_running (started, still up at timeout — a server/CLI) |
    # failed | skipped | none.
    smoke: str = "none"

    @property
    def ok(self) -> bool:
        return (self.status == "verified"
                and self.tests in ("passed", "none")
                and self.smoke in ("passed", "long_running", "none",
                                   "skipped"))

    def repair_feedback(self) -> str:
        parts = [f"{f['file']}: {f['error']}" for f in self.failures[:10]]
        if self.tests == "failed":
            parts.append("TEST FAILURES:\n" + self.test_output[-2000:])
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {"status": self.status, "checked": self.checked,
                "failures": list(self.failures), "skipped": list(self.skipped),
                "tests": self.tests, "backend": self.backend,
                "stack": self.stack, "smoke": self.smoke}

    def report_text(self) -> str:
        lines = ["ZapTheTrick sandbox verification report",
                 "=" * 42,
                 f"status : {self.status} (backend: {self.backend or 'n/a'})",
                 f"checked: {self.checked} file(s)",
                 f"tests  : {self.tests}",
                 f"smoke  : {self.smoke} (entrypoint executed in sandbox)"]
        if self.stack:
            lines.append(f"stack  : {self.stack.get('language')}"
                         f" / {self.stack.get('framework') or '-'}")
        for f in self.failures:
            lines.append(f"FAIL   : {f['file']}: {f['error']}")
        for s in self.skipped:
            lines.append(f"skip   : {s}")
        if self.tests == "failed" and self.test_output:
            lines += ["", "--- test output (tail) ---",
                      self.test_output[-1500:]]
        return "\n".join(lines) + "\n"


def files_from_zip(data: bytes) -> dict[str, str]:
    """Text members of a generated project archive (size/count capped)."""
    out: dict[str, str] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist()[:_MAX_FILES]:
                if info.is_dir() or info.file_size > _MAX_FILE_BYTES:
                    continue
                raw = zf.read(info)
                try:
                    out[info.filename] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue          # binary member — not verifiable text
    except Exception:  # noqa: BLE001
        return {}
    return out


def _stage(files: dict[str, str]) -> str:
    ws = tempfile.mkdtemp(prefix="dtt-verify-")
    root = pathlib.Path(ws)
    for rel, content in list(files.items())[:_MAX_FILES]:
        p = root / pathlib.PurePosixPath(rel)
        try:
            p.relative_to(root)
        except ValueError:
            continue                   # path escape — refuse to stage
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return ws


def _has_tests(files: dict[str, str]) -> bool:
    return any(("test" in pathlib.PurePosixPath(n).name.lower()
                and n.endswith(".py")) for n in files)


def _entrypoint(files: dict[str, str]) -> str | None:
    """The file to smoke-run: a conventional entrypoint at the shallowest
    depth, else the project's single non-test .py file."""
    by_name: dict[str, str] = {}
    for n in files:
        base = pathlib.PurePosixPath(n).name.lower()
        prev = by_name.get(base)
        if prev is None or n.count("/") < prev.count("/"):
            by_name[base] = n
    for cand in _ENTRYPOINTS:
        if cand in by_name:
            return by_name[cand]
    pys = [n for n in files
           if n.endswith(".py") and "test" not in n.lower()]
    return pys[0] if len(pys) == 1 else None


def _module_in_project(module: str, files: dict[str, str]) -> bool:
    top = module.split(".")[0]
    return any(n == f"{top}.py" or n.startswith(f"{top}/")
               or n.endswith(f"/{top}.py") for n in files)


def _install_deps(ws: str, files: dict[str, str],
                  v: ProjectVerification) -> dict | None:
    """Opt-in dependency tier: pip-install requirements.txt into a local
    `_dtt_deps` target inside the sandbox (network-allowed run), so the smoke
    run and tests execute against REAL dependencies. Returns the extra env
    (PYTHONPATH) or None."""
    if "requirements.txt" not in files:
        return None
    import sys

    from app.sandbox import SandboxLimits, run_command
    limits = SandboxLimits.from_config()
    limits.timeout_s = 180.0
    limits.max_files_mb = max(limits.max_files_mb, 256)
    limits.memory_mb = max(limits.memory_mb, 1024)
    res = run_command(
        [sys.executable or "python3", "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", "--no-input",
         "--target", "_dtt_deps", "-r", "requirements.txt"],
        workdir=ws, limits=limits, allow_network=True)
    if res.ok:
        v.skipped.append("deps: requirements.txt installed for smoke/tests")
        return {"PYTHONPATH": "_dtt_deps"}
    v.skipped.append("deps: install failed — "
                     + (res.stderr or res.reason or "")[-200:])
    return None


def _fold_test_result(v: ProjectVerification, res, label: str) -> None:
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    v.test_output = (v.test_output + f"\n\n[{label}]\n" + out).strip()
    if res.status == "unavailable":
        if v.tests == "none":
            v.tests = "skipped"
    elif res.ok:
        if v.tests in ("none", "skipped"):
            v.tests = "passed"
    else:
        v.tests = "failed"
        v.status = "failed"


def _run_other_tests(ws: str, files: dict[str, str], v: ProjectVerification,
                     t_timeout: float, extra_env: dict | None = None) -> None:
    """Non-Python test suites (2026-07-09): node's BUILT-IN test runner for
    *.test.js/*.test.mjs/*.spec.js (no npm install needed) and `go test`
    when the module's deps are vendored. Absent toolchains are recorded as
    skips, never blessed."""
    import shutil as _sh

    from app.sandbox import SandboxLimits, run_command

    js_tests = [n for n in files
                if n.endswith((".test.js", ".test.mjs", ".spec.js"))]
    if js_tests:
        if _sh.which("node"):
            limits = SandboxLimits.from_config()
            limits.timeout_s = t_timeout
            r = run_command(["node", "--test"], workdir=ws, limits=limits,
                            extra_env=extra_env)
            _fold_test_result(v, r, "node --test")
        else:
            v.skipped.append("js tests present but node unavailable")

    go_tests = [n for n in files if n.endswith("_test.go")]
    if go_tests:
        has_gomod = any(n.split("/")[-1] == "go.mod" for n in files)
        vendored = any(n.startswith("vendor/") or "/vendor/" in n
                       for n in files)
        if not _sh.which("go"):
            v.skipped.append("go tests present but go toolchain unavailable")
        elif has_gomod and not vendored:
            v.skipped.append(
                "go tests skipped: deps not vendored (no network in sandbox)")
        else:
            limits = SandboxLimits.from_config()
            limits.timeout_s = max(t_timeout, 120.0)
            limits.max_files_mb = max(limits.max_files_mb, 256)
            r = run_command(["go", "test", "./..."], workdir=ws,
                            limits=limits,
                            extra_env={**(extra_env or {}),
                                       "GOFLAGS": "-mod=vendor"
                                       if vendored else ""})
            _fold_test_result(v, r, "go test")


def _smoke_run(ws: str, files: dict[str, str], v: ProjectVerification,
               extra_env: dict | None = None) -> None:
    """Actually EXECUTE the project's entrypoint in the sandbox (user ask:
    "ensure it compiles and executes successfully"). Catches the class of
    error py_compile can't: import-time crashes, NameErrors at module level,
    bad top-level wiring. A program still running at the timeout (server,
    interactive CLI) counts as started. Missing THIRD-PARTY deps are recorded
    as a skip, not a failure — the sandbox doesn't pip-install."""
    import re as _re
    import sys

    from app.sandbox import SandboxLimits, run_command

    entry = _entrypoint(files)
    if entry is None:
        return
    if entry.endswith(".py"):
        argv = [sys.executable or "python3", "-I", "-B", entry]
    else:
        import shutil as _sh
        node = _sh.which("node")
        if not node:
            v.smoke = "skipped"
            v.skipped.append(f"smoke run: no node for {entry}")
            return
        argv = [node, entry]
    limits = SandboxLimits.from_config()
    limits.timeout_s = min(limits.timeout_s, 8.0)
    res = run_command(argv, workdir=ws, limits=limits, extra_env=extra_env)
    if res.status == "timeout":
        v.smoke = "long_running"        # started and kept running — a server
    elif res.status == "unavailable":
        v.smoke = "skipped"
    elif res.ok:
        v.smoke = "passed"
    else:
        err = (res.stderr or res.stdout or res.reason or "")[-800:]
        if "EOFError" in err:           # interactive CLI hit closed stdin
            v.smoke = "long_running"
            return
        m = _re.search(
            r"(?:ModuleNotFoundError|ImportError): No module named "
            r"['\"]([\w.]+)['\"]", err)
        if m and not _module_in_project(m.group(1), files):
            v.smoke = "skipped"
            v.skipped.append(
                f"smoke run: third-party dependency '{m.group(1)}' "
                "not installed in sandbox")
        else:
            v.smoke = "failed"
            v.status = "failed"
            v.failures.append({
                "file": entry,
                "error": "runtime error on launch: " + err[:400]})


def verify_project_files(files: dict[str, str]) -> ProjectVerification:
    """Stage → detect stack → run the check driver (and tests) in the
    sandbox. Never raises; empty/unstageable input → status 'skipped'."""
    v = ProjectVerification()
    if not files:
        return v
    ws = _stage(files)
    try:
        import sys

        from app.sandbox import SandboxLimits, run_command

        # Stack detection feeds the report + which checks make sense.
        try:
            from app.codeintel.stack_profile import detect_stack
            sp = detect_stack(ws)
            v.stack = {"language": sp.language, "framework": sp.framework}
        except Exception:  # noqa: BLE001
            v.stack = None

        driver = pathlib.Path(ws, "_dtt_verify_driver.py")
        driver.write_text(_DRIVER, encoding="utf-8")
        res = run_command([sys.executable or "python3", "-I", "-B",
                           "_dtt_verify_driver.py"], workdir=ws)
        v.backend = res.backend
        if res.status == "unavailable":
            v.status = "skipped"
            v.skipped.append(f"sandbox unavailable: {res.reason}")
            return v
        try:
            verdict = json.loads((res.stdout or "").strip().splitlines()[-1])
            v.checked = int(verdict.get("checked", 0))
            v.failures = list(verdict.get("failures", []))
            v.skipped = list(verdict.get("skipped", []))
        except Exception:  # noqa: BLE001 — driver crashed → that IS a failure
            v.failures = [{"file": "_driver_",
                           "error": (res.stderr or res.reason or
                                     "verification driver failed")[:400]}]

        # Non-checkable languages are reported, not silently blessed.
        lang = (v.stack or {}).get("language") or ""
        if lang and lang not in ("python", "javascript", "typescript") \
                and v.checked == 0:
            v.skipped.append(f"{lang}: no toolchain for syntax check")

        v.status = ("failed" if v.failures
                    else ("partial" if v.skipped and not v.checked
                          else "verified"))

        # Tests: run the project's own pytest suite in the sandbox.
        try:
            from app.core.config_loader import cfg
            run_tests = bool(getattr(cfg.artifact_validation, "run_tests", True))
            t_timeout = float(getattr(cfg.artifact_validation,
                                      "test_timeout_s", 60.0))
        except Exception:  # noqa: BLE001
            run_tests, t_timeout = True, 60.0
        # Opt-in dependency tier: install requirements.txt in the sandbox so
        # smoke/tests run against real deps (network-allowed run).
        _env: dict | None = None
        try:
            from app.core.config_loader import cfg as _cfgd
            if bool(getattr(_cfgd.artifact_validation, "install_deps",
                            False)) and not v.failures:
                _env = _install_deps(ws, files, v)
        except Exception:  # noqa: BLE001
            _env = None

        if run_tests and not v.failures and _has_tests(files):
            # Keep the CONFIGURED cpu/memory/output limits — only the
            # timeout differs for a test run (a bare SandboxLimits(...) was
            # silently resetting them to dataclass defaults).
            _tl = SandboxLimits.from_config()
            _tl.timeout_s = t_timeout
            tres = run_command(
                [sys.executable or "python3", "-I", "-B", "-m", "pytest",
                 "-q", "--no-header", "-x"],
                workdir=ws, limits=_tl, extra_env=_env)
            v.test_output = (tres.stdout + "\n" + tres.stderr).strip()
            if tres.status == "unavailable":
                v.tests = "skipped"
            elif tres.ok or "no tests ran" in v.test_output:
                v.tests = "passed" if tres.ok else "none"
            else:
                v.tests = "failed"
                v.status = "failed"

        # Non-Python test suites (node --test / vendored go test).
        if run_tests and not v.failures:
            try:
                _run_other_tests(ws, files, v, t_timeout, extra_env=_env)
            except Exception:  # noqa: BLE001 — extra runners never break
                pass

        # Smoke run: execute the entrypoint (config-gated; syntax must be
        # clean first — a compile failure already tells the whole story).
        try:
            from app.core.config_loader import cfg as _cfg
            _smoke_on = bool(getattr(_cfg.artifact_validation,
                                     "smoke_run", True))
        except Exception:  # noqa: BLE001
            _smoke_on = True
        if _smoke_on and not v.failures:
            _smoke_run(ws, files, v, extra_env=_env)
        return v
    except Exception as exc:  # noqa: BLE001
        v.status = "skipped"
        v.skipped.append(f"verifier error: {exc}")
        return v
    finally:
        shutil.rmtree(ws, ignore_errors=True)


__all__ = ["ProjectVerification", "verify_project_files", "files_from_zip"]
