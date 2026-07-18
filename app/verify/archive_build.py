"""Archive creation as a SANDBOX-EXECUTED script (user ask 2026-07-09: "when
I ask to archive the project it should again use sandbox to write a script to
do archive generation carefully").

Instead of zipping in-process, the project files are staged into an ephemeral
workspace and a small stdlib script is EXECUTED INSIDE the sandbox to build
the archive — same isolation as project verification. The resulting bytes are
read back and (in /api/documents/export) still go through the sandbox verify
loop. Fail-open: any sandbox problem returns None and the caller falls back
to the in-process builder, so a download never breaks.
"""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile

log = logging.getLogger(__name__)

_ZIP_DRIVER = r"""
import os, zipfile
OUT = "_dtt_archive_out.zip"
SELF = "_dtt_build_archive.py"
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "__pycache__", "node_modules")]
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(root, n), ".")
            rel = rel.replace(os.sep, "/")
            if rel in (OUT, SELF):
                continue
            zf.write(os.path.join(root, n), rel)
print("OK")
"""

_7Z_DRIVER = r"""
import os
try:
    import py7zr
except Exception:
    print("NO7Z")
    raise SystemExit(0)
OUT = "_dtt_archive_out.7z"
SELF = "_dtt_build_archive.py"
with py7zr.SevenZipFile(OUT, "w") as z:
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "__pycache__", "node_modules")]
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(root, n), ".")
            rel = rel.replace(os.sep, "/")
            if rel in (OUT, SELF):
                continue
            z.write(os.path.join(root, n), rel)
print("OK")
"""


def build_archive_sandboxed(readme: str, files: list[tuple[str, str]],
                            fmt: str = "zip") -> bytes | None:
    """Stage the project files, run the archive-builder script in the
    sandbox, and return the archive bytes. None → caller falls back to the
    in-process builder (sandbox off/unavailable, py7zr missing, any error)."""
    try:
        from app.documents.generators import _safe_zip_name
        from app.sandbox import SandboxLimits, run_command
    except Exception:  # noqa: BLE001
        return None

    driver = _7Z_DRIVER if fmt == "7z" else _ZIP_DRIVER
    out_name = f"_dtt_archive_out.{'7z' if fmt == '7z' else 'zip'}"
    ws = tempfile.mkdtemp(prefix="dtt-archive-")
    try:
        root = pathlib.Path(ws)
        used: set[str] = set()
        if readme:
            used.add("README.md")
            (root / "README.md").write_text(readme, encoding="utf-8")
        for name, body in files:
            fname = _safe_zip_name(name, used)
            p = root / pathlib.PurePosixPath(fname)
            try:
                p.resolve().relative_to(root.resolve())
            except ValueError:
                continue                    # path escape — refuse to stage
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text((body or "").rstrip("\n") + "\n", encoding="utf-8")
        (root / "_dtt_build_archive.py").write_text(driver, encoding="utf-8")

        import sys
        limits = SandboxLimits.from_config()
        limits.timeout_s = min(max(limits.timeout_s, 20.0), 30.0)
        res = run_command([sys.executable or "python3", "-I", "-B",
                           "_dtt_build_archive.py"], workdir=ws, limits=limits)
        if not res.ok or "OK" not in (res.stdout or ""):
            if "NO7Z" not in (res.stdout or ""):
                log.debug("sandbox archive build fell back: %s/%s",
                          res.status, (res.stderr or res.reason or "")[:120])
            return None
        out = root / out_name
        if not out.is_file():
            return None
        data = out.read_bytes()
        return data or None
    except Exception as exc:  # noqa: BLE001 — never break a download
        log.debug("sandbox archive build error: %s", exc)
        return None
    finally:
        shutil.rmtree(ws, ignore_errors=True)
