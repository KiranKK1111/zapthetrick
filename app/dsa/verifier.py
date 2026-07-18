"""Stage 7 — Verifier (sandboxed code execution).

Python-first per Architecture2.md §"Sandboxed code execution". Other
languages return `skipped=True` until a Pyodide / QuickJS adapter
lands.

Strategy:
  1. Generate driver code that defines the user's function + runs the
     test cases.
  2. Execute it in a fresh `python` subprocess with `-I` (isolated)
     and a wall-clock timeout via `asyncio.subprocess`.
  3. Compare stdout to expected outputs.
  4. On any failure, return `errors` — the pipeline's repair loop
     feeds those back to the solution generator.

Resource limits (CPU / memory) are POSIX-only via `resource.setrlimit`.
On Windows we rely on the wall-clock timeout alone — same level of
safety as a typical CI sandbox.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

from .types import SolutionApproach, VerifyResult


log = logging.getLogger(__name__)


_TIMEOUT_SECONDS = 4.0


async def verify(
    approach: SolutionApproach | None,
    examples: list[dict],
) -> VerifyResult:
    """Run the approach's code against `examples`. Returns counts +
    error messages the repair loop can feed back to the model.

    `examples` items expected shape:
        {"input": "...", "expected_output": "..."}
    The driver script tries to import the user's function; if no
    obvious entry-point exists, it just `exec`s the code and skips.
    """
    if approach is None or not approach.code.strip():
        return VerifyResult(skipped=True)
    if approach.language != "python":
        # JS / Java / Go / C++ — not wired yet. Architecture2.md tags
        # these as "unverified" in the badge.
        return VerifyResult(skipped=True)
    if not examples:
        return VerifyResult(skipped=True)

    driver = _build_driver(approach.code, examples)
    try:
        rc, stdout, stderr = await _run_python(driver)
    except FileNotFoundError:
        return VerifyResult(skipped=True, errors=["python interpreter not found in PATH"])
    except asyncio.TimeoutError:
        return VerifyResult(skipped=False, failed=len(examples), errors=["timeout"])

    return _parse_driver_output(stdout, stderr, expected_count=len(examples), rc=rc)


def _build_driver(user_code: str, examples: list[dict]) -> str:
    """Wrap the user's code with a harness that runs each example +
    prints a single `RESULT i: PASS|FAIL <details>` line per test.

    The harness is intentionally simple: it tries to find a single
    top-level function and call it with the example's `input` as a
    Python literal. If the input isn't valid Python (or no function
    is found), the test is marked `SKIP`.
    """
    import json as _json

    cases_json = _json.dumps(examples)
    return f"""
import io
import ast
import sys
import traceback
import json as _json

USER_CODE = {user_code!r}
CASES = _json.loads({cases_json!r})

# Exec user code into its own namespace.
_ns: dict = {{}}
try:
    exec(USER_CODE, _ns)
except Exception as exc:  # noqa
    print(f"SETUP_FAIL: {{type(exc).__name__}}: {{exc}}")
    sys.exit(0)

# Find a callable to test against — prefer the first user-defined function.
candidates = [
    (k, v) for k, v in _ns.items()
    if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) in (None, "__main__")
]
if not candidates:
    # No function — maybe the user wrote a class. Skip.
    print("DRIVER_SKIP: no top-level function found")
    sys.exit(0)
_fn_name, _fn = candidates[0]

def _coerce(raw: str):
    # Try Python literal; fall back to plain string.
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw

def _call_with(arg):
    # If it's a tuple, splat. Otherwise pass as single arg.
    if isinstance(arg, tuple):
        return _fn(*arg)
    return _fn(arg)

for i, case in enumerate(CASES):
    raw_in = str(case.get("input", ""))
    raw_expected = str(case.get("expected_output", "")).strip()
    try:
        parsed = _coerce(raw_in)
        actual = _call_with(parsed)
        actual_s = repr(actual) if not isinstance(actual, str) else actual
        expected_parsed = _coerce(raw_expected)
        expected_s = repr(expected_parsed) if not isinstance(expected_parsed, str) else expected_parsed
        ok = (
            actual == expected_parsed
            or str(actual).strip() == str(expected_parsed).strip()
        )
        if ok:
            print(f"RESULT {{i}}: PASS")
        else:
            print(f"RESULT {{i}}: FAIL  expected={{expected_s}}  got={{actual_s}}")
    except Exception as exc:  # noqa
        tb = traceback.format_exception_only(type(exc), exc)
        print(f"RESULT {{i}}: ERROR  {{''.join(tb).strip()}}")
"""


async def _run_python(driver: str) -> tuple[int, str, str]:
    """Run driver in a subprocess. Wall-clock timeout via
    asyncio.wait_for; POSIX resource limits via preexec_fn when
    available."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(driver)
        path = Path(f.name)

    try:
        preexec = _posix_resource_limits if os.name == "posix" else None
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",                                  # ignore PYTHON* env, no user site
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise
        return proc.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _posix_resource_limits():  # noqa: ANN201 — preexec_fn signature
    """Cap CPU + virtual memory on POSIX. No-op on Windows."""
    import resource  # type: ignore

    # 5 seconds of CPU time.
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    # 512 MB virtual address space.
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))


def _parse_driver_output(stdout: str, stderr: str, *, expected_count: int, rc: int) -> VerifyResult:
    passed = 0
    failed = 0
    errors: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("RESULT") and ":" in line:
            tag, _, detail = line.partition(":")
            detail = detail.strip()
            if detail.startswith("PASS"):
                passed += 1
            else:
                failed += 1
                errors.append(f"{tag.strip()}: {detail}")
        elif line.startswith("SETUP_FAIL") or line.startswith("DRIVER_SKIP"):
            errors.append(line.strip())
    if stderr.strip() and not errors:
        errors.append(f"stderr: {stderr.strip()[:400]}")
    # If we got fewer RESULT lines than cases, treat the missing ones as failed.
    counted = passed + failed
    if counted < expected_count:
        failed += expected_count - counted
        errors.append(f"only {counted}/{expected_count} cases produced a result line")
    return VerifyResult(
        passed=passed, failed=failed, errors=errors[:8], skipped=(counted == 0 and not errors)
    )
