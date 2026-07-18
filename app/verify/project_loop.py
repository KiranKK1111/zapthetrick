"""The closed project loop for CHAT builds: verify → model-repair → re-verify
→ package with an honest verification report (the Claude behavior the user
asked for — a downloaded project has actually been checked and tested in the
sandbox, not just emitted).

`verify_and_repair_archive(zip_bytes)`:
  1. parse the generated archive's text members;
  2. run [verify_project_files] in the dedicated sandbox (syntax + JSON/YAML
     + the project's own pytest suite);
  3. on failure (and when `artifact_validation.repair_with_model`): ONE
     bounded LLM repair round — the model receives the failing files + the
     compiler/test output and returns corrected files; re-verify;
  4. rebuild the archive with the (possibly corrected) files and a
     `VERIFICATION.txt` report — verified or not, the user sees the truth.

Fail-open at every stage: any error returns the ORIGINAL bytes untouched
(plus whatever report exists).
"""
from __future__ import annotations

import io
import logging
import re
import zipfile

from .project_verify import (ProjectVerification, files_from_zip,
                             verify_project_files)

log = logging.getLogger(__name__)

_REPAIR_SYSTEM = (
    "You fix broken generated projects. You get project files and the "
    "compiler/test errors from a sandbox run. Return ONLY the corrected "
    "files that need changes, each as a fenced code block whose info line is "
    "the exact file path, e.g.:\n```app/main.py\n<full corrected content>\n"
    "```\nReturn complete file contents (no diffs, no commentary).")

# ```path/to/file.ext\n...content...\n```  (the info string is the path)
_FENCE_RE = re.compile(
    r"```([\w./\\-]+\.[A-Za-z0-9]+)[ \t]*\n(.*?)```", re.DOTALL)


def parse_fenced_files(text: str) -> dict[str, str]:
    """Corrected files from the repair model's reply."""
    out: dict[str, str] = {}
    for m in _FENCE_RE.finditer(text or ""):
        path = m.group(1).strip().replace("\\", "/")
        if path and ".." not in path.split("/"):
            out[path] = m.group(2)
    return out


async def _model_repair(files: dict[str, str],
                        verification: ProjectVerification) -> dict[str, str]:
    """One bounded repair round. Returns the corrected-file map ({} = no
    usable repair)."""
    try:
        from app.core.llm_client import llm
        failing = {f["file"] for f in verification.failures}
        # Send the failing files (plus a bounded slice of the rest for
        # context) — not the whole project.
        shown, budget = [], 24_000
        for name in list(failing) + [n for n in files if n not in failing]:
            content = files.get(name)
            if content is None or budget <= 0:
                continue
            snippet = content[:6_000]
            shown.append(f"```{name}\n{snippet}\n```")
            budget -= len(snippet)
        user = ("The sandbox verification failed.\n\nERRORS:\n"
                + verification.repair_feedback()[:4_000]
                + "\n\nPROJECT FILES:\n" + "\n".join(shown)
                + "\n\nReturn the corrected files now.")
        reply = await llm.complete_routed(
            [{"role": "system", "content": _REPAIR_SYSTEM},
             {"role": "user", "content": user}],
            options={"purpose": "repair"})
        text = reply if isinstance(reply, str) else str(reply or "")
        return parse_fenced_files(text)
    except Exception as exc:  # noqa: BLE001 — no repair beats a broken turn
        log.info("project repair round skipped: %s", exc)
        return {}


def files_from_7z(data: bytes) -> dict[str, str]:
    """Text members of a generated .7z project (same caps as the zip
    reader). Empty dict when py7zr is missing or the archive is unreadable."""
    out: dict[str, str] = {}
    try:
        import py7zr
        with py7zr.SevenZipFile(io.BytesIO(data)) as z:
            for name, bio in (z.readall() or {}).items():
                if len(out) >= 400:
                    break
                raw = bio.read()
                if len(raw) > 512 * 1024:
                    continue
                try:
                    out[name] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
    except Exception:  # noqa: BLE001
        return {}
    return out


def _rebuild_7z(files: dict[str, str], original: bytes,
                report: str) -> bytes:
    """Mirror of _rebuild_zip for .7z exports: corrected text members +
    VERIFICATION.txt; binary members carried over untouched."""
    try:
        import py7zr
        members: dict[str, bytes] = {}
        with py7zr.SevenZipFile(io.BytesIO(original)) as src:
            for name, bio in (src.readall() or {}).items():
                members[name] = bio.read()
        for name, content in files.items():
            members[name] = content.encode("utf-8")
        members["VERIFICATION.txt"] = report.encode("utf-8")
        buf = io.BytesIO()
        with py7zr.SevenZipFile(buf, "w") as dst:
            for name, raw in members.items():
                dst.writestr(raw, name)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return original


def _rebuild_zip(files: dict[str, str], original: bytes,
                 report: str) -> bytes:
    """Original archive rebuilt with (possibly corrected) text members +
    VERIFICATION.txt. Binary members are carried over untouched."""
    buf = io.BytesIO()
    try:
        with zipfile.ZipFile(io.BytesIO(original)) as src, \
                zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                if info.is_dir():
                    continue
                if info.filename in files:
                    dst.writestr(info.filename, files[info.filename])
                else:
                    dst.writestr(info.filename, src.read(info))
            for name, content in files.items():   # repair may ADD files
                if name not in src.namelist():
                    dst.writestr(name, content)
            dst.writestr("VERIFICATION.txt", report)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return original


async def verify_and_repair_archive(
        data: bytes, fmt: str = "zip") -> tuple[bytes, dict | None]:
    """The chat-download choke point: (possibly repaired) archive bytes +
    the verification meta (None when the feature is off / not a project).
    `fmt` is "zip" (default) or "7z" — both are verified the same way."""
    try:
        from app.core.config_loader import cfg
        av = cfg.artifact_validation
        if not bool(getattr(av, "verify_projects", True)):
            return data, None
        repair_on = bool(getattr(av, "repair_with_model", True))
    except Exception:  # noqa: BLE001
        repair_on = True

    _read = files_from_7z if fmt == "7z" else files_from_zip
    _rebuild = _rebuild_7z if fmt == "7z" else _rebuild_zip
    try:
        files = _read(data)
        if not files:
            return data, None
        import asyncio
        v = await asyncio.to_thread(verify_project_files, files)
        # Bounded repair LOOP (was a single round): each round feeds the
        # latest sandbox errors back to the router — which naturally rotates
        # models between rounds, so a stuck fix gets a second opinion.
        try:
            from app.core.config_loader import cfg as _cfg2
            rounds = int(getattr(_cfg2.artifact_validation,
                                 "repair_rounds", 2))
        except Exception:  # noqa: BLE001
            rounds = 2
        repairs = 0
        for _ in range(max(0, rounds if repair_on else 0)):
            if v.status != "failed":
                break
            fixes = await _model_repair(files, v)
            if not fixes:
                break
            files.update(fixes)
            v2 = await asyncio.to_thread(verify_project_files, files)
            if v2.status == "skipped":
                break
            v = v2
            repairs += 1
        repaired = repairs > 0
        meta = v.as_dict()
        meta["repaired"] = repaired
        meta["repair_rounds"] = repairs
        report = v.report_text() + (
            f"\nNOTE: {repairs} automatic repair round(s) were applied.\n"
            if repaired else "")
        out = await asyncio.to_thread(_rebuild, files, data, report)
        return out, meta
    except Exception as exc:  # noqa: BLE001 — never block the download
        log.info("project verification skipped: %s", exc)
        return data, None


__all__ = ["verify_and_repair_archive", "parse_fenced_files"]
