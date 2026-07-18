"""Workspace materializer (Spec §8.4-1) — turn a chat upload into a real,
editable, sandboxed project folder the agent loop can read / edit / run.

  res = materialize_archive(conversation_id, data, "project.zip")
  path = workspace_path(conversation_id)        # reuse on follow-ups
  zip_bytes = package_workspace(conversation_id) # return the modified project
  diff = diff_summary(path)                       # what the agent changed

Safety is the whole point (uploaded code is UNTRUSTED):
  - **zip-slip / path-traversal** guards (no abs paths, no `..`, must resolve
    inside the workspace root);
  - **symlinks skipped** (a symlink member can escape the root);
  - **archive-bomb caps** — max file count, per-file size, total uncompressed
    size; extraction stops and flags `truncated` when a cap is hit;
  - confined to `<WS_ROOT>/<conversation_id>/`.

Supports zip · tar(.gz/.bz2/.xz) · 7z · rar. Pure filesystem + stdlib + the
already-bundled `py7zr` / `rarfile`.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Caps (archive-bomb / disk protection).
MAX_FILES = 20_000
MAX_FILE_BYTES = 60 * 1024 * 1024          # 60 MB single file
MAX_TOTAL_BYTES = 600 * 1024 * 1024        # 600 MB uncompressed total
# Workspace quota: keep at most this many conversation workspaces (LRU evict).
MAX_WORKSPACES = 24

# Dirs we never package back into a download (regeneratable / huge).
_PACKAGE_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                      "dist", "build", "target", ".gradle", ".idea", ".mypy_cache"}


@dataclass
class MaterializeResult:
    path: str
    files: int = 0
    bytes: int = 0
    skipped: int = 0
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.files > 0


# --------------------------------------------------------------------------
# Workspace root + paths
# --------------------------------------------------------------------------
def ws_root() -> str:
    base = os.environ.get("ZAPTHETRICK_WS_ROOT") or os.path.join(
        os.path.expanduser("~"), ".zapthetrick", "agent_workspaces")
    os.makedirs(base, exist_ok=True)
    return base


def _safe_cid(conversation_id: str) -> str:
    cid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(conversation_id or "default"))
    return cid[:80] or "default"


def workspace_path(conversation_id: str) -> str:
    return os.path.join(ws_root(), _safe_cid(conversation_id))


def workspace_exists(conversation_id: str) -> bool:
    p = workspace_path(conversation_id)
    return os.path.isdir(p) and bool(os.listdir(p))


def fresh_workspace(conversation_id: str, *, reset: bool = False) -> str:
    """Return an (optionally emptied) workspace dir for build-from-scratch."""
    p = workspace_path(conversation_id)
    if reset and os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p, exist_ok=True)
    return p


def cleanup(conversation_id: str) -> None:
    shutil.rmtree(workspace_path(conversation_id), ignore_errors=True)


def enforce_quota(max_workspaces: int = MAX_WORKSPACES) -> int:
    """LRU-evict oldest workspaces beyond the cap. Returns how many removed."""
    root = ws_root()
    try:
        entries = [os.path.join(root, d) for d in os.listdir(root)]
        dirs = [d for d in entries if os.path.isdir(d)]
    except OSError:
        return 0
    if len(dirs) <= max_workspaces:
        return 0
    dirs.sort(key=lambda d: os.path.getmtime(d))  # oldest first
    removed = 0
    for d in dirs[: len(dirs) - max_workspaces]:
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Safe extraction
# --------------------------------------------------------------------------
def _safe_target(root_real: str, name: str) -> str | None:
    """Resolve archive member `name` under `root_real`; None if it escapes.
    Members containing a traversal (`..`) or absolute path are rejected
    outright rather than silently flattened — an upload that tries to escape
    is treated as hostile."""
    name = (name or "").replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None  # absolute path
    raw = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in raw):
        return None  # path traversal attempt
    parts = raw
    if not parts:
        return None
    target = os.path.realpath(os.path.join(root_real, *parts))
    if target != root_real and not target.startswith(root_real + os.sep):
        return None
    return target


def _write_member(root_real: str, name: str, data: bytes,
                  res: MaterializeResult) -> bool:
    """Write one member with caps + path guards. Returns False to stop (cap)."""
    if res.files >= MAX_FILES or res.bytes >= MAX_TOTAL_BYTES:
        res.truncated = True
        return False
    if len(data) > MAX_FILE_BYTES:
        res.skipped += 1
        return True
    target = _safe_target(root_real, name)
    if target is None:
        res.skipped += 1  # zip-slip / traversal attempt — dropped
        return True
    os.makedirs(os.path.dirname(target) or root_real, exist_ok=True)
    try:
        with open(target, "wb") as f:
            f.write(data)
    except OSError:
        res.skipped += 1
        return True
    res.files += 1
    res.bytes += len(data)
    return True


def _extract_zip(data: bytes, root_real: str, res: MaterializeResult) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                res.skipped += 1
                continue
            if info.file_size > MAX_FILE_BYTES:
                res.skipped += 1
                continue
            try:
                payload = z.read(info)
            except Exception:  # noqa: BLE001
                res.skipped += 1
                continue
            if not _write_member(root_real, info.filename, payload, res):
                return


def _extract_tar(data: bytes, root_real: str, res: MaterializeResult) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        for m in t:
            if not m.isfile():        # skip dirs, symlinks, hardlinks, devices
                continue
            if m.size > MAX_FILE_BYTES:
                res.skipped += 1
                continue
            f = t.extractfile(m)
            if f is None:
                res.skipped += 1
                continue
            if not _write_member(root_real, m.name, f.read(), res):
                return


def _extract_7z(data: bytes, root_real: str, res: MaterializeResult) -> None:
    # Memory-safe (2026-07-09): py7zr decompresses whole read targets into
    # RAM, so DECLARED sizes are checked BEFORE any decompression (a 7z bomb
    # can't spike memory) and extraction happens in bounded batches instead
    # of one readall() of the entire archive.
    import py7zr

    with py7zr.SevenZipFile(io.BytesIO(data)) as z:
        infos = z.list()
    names: list[str] = []
    total = 0
    for fi in infos:
        if getattr(fi, "is_directory", False):
            continue
        size = int(getattr(fi, "uncompressed", 0) or 0)
        if size > MAX_FILE_BYTES:
            res.skipped += 1
            continue
        if total + size > MAX_TOTAL_BYTES or len(names) >= MAX_FILES:
            res.truncated = True
            break
        total += size
        names.append(fi.filename)
    i = 0
    while i < len(names):
        batch = names[i:i + 200]
        i += len(batch)
        with py7zr.SevenZipFile(io.BytesIO(data)) as z:
            got = z.read(targets=batch) or {}
        for name, bio in got.items():
            payload = bio.read() if hasattr(bio, "read") else bytes(bio or b"")
            if len(payload) > MAX_FILE_BYTES:
                res.skipped += 1
                continue
            if not _write_member(root_real, name, payload, res):
                return


def _extract_rar(data: bytes, root_real: str, res: MaterializeResult) -> None:
    import rarfile

    with rarfile.RarFile(io.BytesIO(data)) as r:
        for info in r.infolist():
            if info.isdir():
                continue
            if info.file_size > MAX_FILE_BYTES:
                res.skipped += 1
                continue
            try:
                payload = r.read(info)
            except Exception:  # noqa: BLE001
                res.skipped += 1
                continue
            if not _write_member(root_real, info.filename, payload, res):
                return


# In-sandbox extractor (user ask 2026-07-09: "read the uploaded archive using
# a sandbox script"): the archive bytes are staged into the target workspace
# and this script is EXECUTED INSIDE the sandbox to extract them — same caps
# and path guards as the in-process extractors (which remain the fail-open
# fallback). rar stays in-process (needs the bundled UnRAR tool).
_EXTRACT_DRIVER = r"""
import io, json, os, stat, sys
# Caps come from the host module (argv) so config/test overrides apply.
MAX_FILES = int(sys.argv[2])
MAX_FILE = int(sys.argv[3])
MAX_TOTAL = int(sys.argv[4])
ARC, SELF = "_dtt_upload_archive.bin", "_dtt_extract.py"
kind = sys.argv[1]
root = os.path.realpath(".")
files = total = skipped = 0
truncated = False

def safe(name):
    n = (name or "").replace("\\", "/")
    if not n or n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return None
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    t = os.path.realpath(os.path.join(root, *parts))
    if t != root and not t.startswith(root + os.sep):
        return None
    return t

def write(name, payload):
    global files, total, skipped, truncated
    if files >= MAX_FILES or total >= MAX_TOTAL:
        truncated = True
        return False
    if len(payload) > MAX_FILE:
        skipped += 1
        return True
    t = safe(name)
    if t is None or os.path.basename(t) in (ARC, SELF):
        skipped += 1
        return True
    os.makedirs(os.path.dirname(t) or root, exist_ok=True)
    with open(t, "wb") as f:
        f.write(payload)
    files += 1
    total += len(payload)
    return True

data = open(ARC, "rb").read()
if kind == "zip":
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                skipped += 1
                continue
            if info.file_size > MAX_FILE:
                skipped += 1
                continue
            if not write(info.filename, z.read(info)):
                break
elif kind == "tar":
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        for m in t:
            if not m.isfile():
                continue
            if m.size > MAX_FILE:
                skipped += 1
                continue
            f = t.extractfile(m)
            if f is None:
                skipped += 1
                continue
            if not write(m.name, f.read()):
                break
elif kind == "7z":
    import py7zr
    with py7zr.SevenZipFile(io.BytesIO(data)) as z:
        infos = z.list()
    names, declared = [], 0
    for fi in infos:
        if getattr(fi, "is_directory", False):
            continue
        size = int(getattr(fi, "uncompressed", 0) or 0)
        if size > MAX_FILE:
            skipped += 1
            continue
        if declared + size > MAX_TOTAL or len(names) >= MAX_FILES:
            truncated = True
            break
        declared += size
        names.append(fi.filename)
    i = 0
    while i < len(names):
        batch = names[i:i + 200]
        i += len(batch)
        with py7zr.SevenZipFile(io.BytesIO(data)) as z:
            got = z.read(targets=batch) or {}
        for name, bio in got.items():
            payload = bio.read() if hasattr(bio, "read") else bytes(bio or b"")
            if not write(name, payload):
                i = len(names)
                break
print(json.dumps({"files": files, "bytes": total, "skipped": skipped,
                  "truncated": truncated}))
"""

_SANDBOX_KINDS = {"_extract_zip": "zip", "_extract_tar": "tar",
                  "_extract_7z": "7z"}


def _extract_sandboxed(data: bytes, kind: str, root_real: str,
                       res: MaterializeResult) -> bool:
    """Extract via the in-sandbox driver. Returns False (and leaves a clean
    slate) when the sandbox is off/unavailable — caller falls back."""
    import json as _json
    import sys

    try:
        from app.sandbox import SandboxLimits, run_command
    except Exception:  # noqa: BLE001
        return False
    arc = os.path.join(root_real, "_dtt_upload_archive.bin")
    drv = os.path.join(root_real, "_dtt_extract.py")
    try:
        with open(arc, "wb") as f:
            f.write(data)
        with open(drv, "w", encoding="utf-8") as f:
            f.write(_EXTRACT_DRIVER)
        limits = SandboxLimits.from_config()
        limits.timeout_s = max(limits.timeout_s, 60.0)
        # FSIZE must allow the per-file cap (POSIX levels), not the 32MB
        # script default.
        limits.max_files_mb = max(limits.max_files_mb, 64)
        cmd = run_command([sys.executable or "python3", "-I", "-B",
                           "_dtt_extract.py", kind, str(MAX_FILES),
                           str(MAX_FILE_BYTES), str(MAX_TOTAL_BYTES)],
                          workdir=root_real, limits=limits)
        if not cmd.ok:
            return False
        verdict = _json.loads((cmd.stdout or "").strip().splitlines()[-1])
        res.files = int(verdict.get("files", 0))
        res.bytes = int(verdict.get("bytes", 0))
        res.skipped = int(verdict.get("skipped", 0))
        res.truncated = bool(verdict.get("truncated", False))
        return res.files > 0
    except Exception as exc:  # noqa: BLE001
        log.debug("sandbox extraction fell back: %s", exc)
        return False
    finally:
        for p in (arc, drv):
            try:
                os.remove(p)
            except OSError:
                pass


def _dispatch(filename: str):
    n = (filename or "").lower()
    if n.endswith(".zip"):
        return _extract_zip
    if n.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
                   ".tar.xz", ".txz")):
        return _extract_tar
    if n.endswith(".7z"):
        return _extract_7z
    if n.endswith(".rar"):
        return _extract_rar
    return None


def materialize_archive(conversation_id: str, data: bytes,
                        filename: str) -> MaterializeResult:
    """Extract an uploaded archive into the conversation's workspace, safely.
    Replaces any prior contents for that conversation."""
    extractor = _dispatch(filename)
    path = fresh_workspace(conversation_id, reset=True)
    res = MaterializeResult(path=path)
    if extractor is None:
        res.error = f"unsupported archive type: {filename}"
        return res
    root_real = os.path.realpath(path)
    # Preferred path: extraction runs as a SCRIPT INSIDE THE SANDBOX (same
    # caps + traversal guards); the in-process extractors below are the
    # fail-open fallback so an upload never breaks.
    _kind = _SANDBOX_KINDS.get(extractor.__name__)
    if _kind is not None and _extract_sandboxed(data, _kind, root_real, res):
        enforce_quota()
        return res
    if res.files or res.skipped:        # partial sandbox attempt → clean slate
        path = fresh_workspace(conversation_id, reset=True)
        root_real = os.path.realpath(path)
        res = MaterializeResult(path=path)
    try:
        extractor(data, root_real, res)
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        log.info("materialize failed for %s: %s", filename, exc)
    enforce_quota()
    return res


# --------------------------------------------------------------------------
# Package back + diff
# --------------------------------------------------------------------------
def package_workspace(conversation_id_or_path: str) -> bytes:
    """Zip the workspace for download (skipping regeneratable/huge dirs)."""
    path = (conversation_id_or_path
            if os.path.isdir(conversation_id_or_path)
            else workspace_path(conversation_id_or_path))
    buf = io.BytesIO()
    root_real = os.path.realpath(path)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dp, dns, fns in os.walk(root_real):
            dns[:] = [d for d in dns if d not in _PACKAGE_SKIP_DIRS]
            for fn in fns:
                fp = os.path.join(dp, fn)
                rel = os.path.relpath(fp, root_real).replace("\\", "/")
                try:
                    zf.write(fp, rel)
                except OSError:
                    continue
    return buf.getvalue()


async def git_init_baseline(path: str) -> None:
    """Snapshot the freshly-materialized tree so we can diff agent edits later.
    Best-effort — silently no-ops if git is unavailable."""
    from .runner import run_in_workspace

    cmds = [
        "git init -q",
        'git -c user.email=agent@zapthetrick -c user.name=ZapTheTrick '
        'add -A',
        'git -c user.email=agent@zapthetrick -c user.name=ZapTheTrick '
        'commit -q -m baseline --allow-empty',
    ]
    for c in cmds:
        await run_in_workspace(c, cwd=path, timeout=60)


async def diff_summary(path: str) -> str:
    """Return a human-readable summary of what changed since the baseline."""
    from .runner import run_in_workspace

    await run_in_workspace("git add -A", cwd=path, timeout=60)
    stat_r = await run_in_workspace(
        "git diff --cached --stat HEAD", cwd=path, timeout=60)
    names_r = await run_in_workspace(
        "git diff --cached --name-status HEAD", cwd=path, timeout=60)
    out = (names_r.stdout or "").strip()
    stat_out = (stat_r.stdout or "").strip()
    if not out and not stat_out:
        return "No file changes."
    return (f"Changed files:\n{out}\n\n{stat_out}").strip()


async def semantic_change_summary(path: str, *, max_files: int = 12) -> list[str]:
    """AST-level summary of what changed since the baseline (#106): per changed
    code file, which symbols were added / removed / had their signature changed.
    Call AFTER `diff_summary` (which stages the tree). Best-effort → []."""
    from app.codegraph.semantic_diff import semantic_diff, summarize_semantic_diff
    from app.codegraph.tsutil import language_for

    from .runner import run_in_workspace

    try:
        names_r = await run_in_workspace(
            "git diff --cached --name-only HEAD", cwd=path, timeout=60)
    except Exception:  # noqa: BLE001
        return []
    files = [f.strip() for f in (names_r.stdout or "").splitlines() if f.strip()]
    out: list[str] = []
    for rel in files[:max_files]:
        if language_for(rel) is None:
            continue
        try:
            old_r = await run_in_workspace(
                f'git show "HEAD:{rel}"', cwd=path, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        old_src = old_r.stdout if (old_r.ok and not old_r.denied) else ""
        target = _safe_target(os.path.realpath(path), rel)
        if target is None or not os.path.isfile(target):
            continue
        try:
            with open(target, encoding="utf-8", errors="replace") as fh:
                new_src = fh.read()
        except OSError:
            continue
        s = summarize_semantic_diff(
            semantic_diff(old_src, new_src, path=rel), path=rel)
        if s:
            out.append(s)
    return out


__all__ = [
    "MaterializeResult", "ws_root", "workspace_path", "workspace_exists",
    "fresh_workspace", "cleanup", "enforce_quota", "materialize_archive",
    "package_workspace", "git_init_baseline", "diff_summary",
]
