"""Run the language's auto-formatter on a code snippet.

Best-effort — when the toolchain isn't installed, the original
text is returned unchanged. Each formatter is invoked via
subprocess so we don't have to depend on the Python bindings.

The formatter selection table is small on purpose: only the ones
where the binary's stdin/stdout contract is clean and the install
is plausible on a dev machine.
"""
from __future__ import annotations

import asyncio
import logging
import shutil


log = logging.getLogger(__name__)


# (binary, args). The formatter must accept code on stdin and emit
# the formatted code on stdout. Non-stdin formatters require a temp
# file roundtrip — out of scope for the initial scaffold.
_FORMATTERS: dict[str, tuple[str, list[str]]] = {
    "python":      ("black",   ["-q", "-"]),
    "javascript":  ("prettier", ["--stdin-filepath", "snippet.js"]),
    "typescript":  ("prettier", ["--stdin-filepath", "snippet.ts"]),
    "go":          ("gofmt",   []),
    "rust":        ("rustfmt", ["--emit=stdout"]),
}


async def format_code(language: str, code: str, *, timeout_s: float = 5.0) -> str:
    """Format `code` in-place. Returns the original on any failure."""
    fmt = _FORMATTERS.get((language or "").lower())
    if not fmt:
        return code
    binary, args = fmt
    if shutil.which(binary) is None:
        return code
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=code.encode("utf-8")), timeout=timeout_s
        )
        if proc.returncode != 0:
            log.info("formatter %s returned %s: %s", binary, proc.returncode, stderr.decode("utf-8", "replace")[:200])
            return code
        return stdout.decode("utf-8", "replace") or code
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
        log.info("formatter %s skipped: %s", binary, exc)
        return code


__all__ = ["format_code"]
