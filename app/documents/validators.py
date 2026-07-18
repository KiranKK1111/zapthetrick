"""Generated-artifact validation + repair + degrade (ArchitectureVerdict
Phase 4 — the doc's Validation Intelligence: "validate generated PDF/DOCX/
PPTX/ZIP", "don't return outputs until they're verified").

Previously `render_document` returned raw bytes straight to the client — a
corrupt render shipped silently. This module closes the loop:

    render → validate → (invalid? re-render once = repair) →
    (still invalid? DEGRADE along the capability fallback chain: pdf→docx→md)

Validation methods are structural and cheap (magic bytes, OpenXML/zip
integrity, parser round-trips) — never an LLM. Formats we can't check are
reported `skipped`, not failed. Everything is fail-open: a validator crash
counts as skipped so artifact delivery is never blocked by its own guard.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

_OPENXML = {"docx", "pptx", "xlsx"}


@dataclass
class ValidationResult:
    ok: bool
    fmt: str
    method: str = ""            # how it was checked (or "skipped")
    reason: str = ""            # failure detail

    def as_dict(self) -> dict:
        return {"ok": self.ok, "format": self.fmt, "method": self.method,
                "reason": self.reason}


def _valid_pdf(data: bytes) -> ValidationResult:
    if not data.startswith(b"%PDF-"):
        return ValidationResult(False, "pdf", "magic", "missing %PDF header")
    try:
        import importlib.util
        if importlib.util.find_spec("pypdf") is not None:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) < 1:
                return ValidationResult(False, "pdf", "pypdf", "zero pages")
            return ValidationResult(True, "pdf", "pypdf")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, "pdf", "pypdf", str(exc)[:120])
    # No pypdf → structural check: a trailer marker near the end.
    ok = b"%%EOF" in data[-2048:]
    return ValidationResult(ok, "pdf", "trailer",
                            "" if ok else "missing %%EOF trailer")


def _valid_zip_container(data: bytes, fmt: str) -> ValidationResult:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            bad = zf.testzip()
            if bad is not None:
                return ValidationResult(False, fmt, "zip", f"corrupt: {bad}")
            if fmt in _OPENXML and "[Content_Types].xml" not in zf.namelist():
                return ValidationResult(False, fmt, "openxml",
                                        "missing [Content_Types].xml")
        return ValidationResult(True, fmt, "zip")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, fmt, "zip", str(exc)[:120])


def _valid_7z(data: bytes) -> ValidationResult:
    try:
        import importlib.util
        if importlib.util.find_spec("py7zr") is None:
            return ValidationResult(True, "7z", "skipped", "py7zr unavailable")
        import py7zr
        with py7zr.SevenZipFile(io.BytesIO(data)) as z:
            z.test()
        return ValidationResult(True, "7z", "py7zr")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, "7z", "py7zr", str(exc)[:120])


def _valid_json(data: bytes) -> ValidationResult:
    try:
        json.loads(data.decode("utf-8"))
        return ValidationResult(True, "json", "parse")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, "json", "parse", str(exc)[:120])


def _valid_csv(data: bytes) -> ValidationResult:
    try:
        text = data.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text)))
        return ValidationResult(bool(rows), "csv", "parse",
                                "" if rows else "no rows")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, "csv", "parse", str(exc)[:120])


def _valid_text(data: bytes, fmt: str) -> ValidationResult:
    try:
        data.decode("utf-8")
        return ValidationResult(True, fmt, "utf8")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, fmt, "utf8", str(exc)[:120])


def validate_artifact(data: bytes, fmt: str) -> ValidationResult:
    """Structural validation of a generated artifact. Unknown formats are
    `skipped` (ok=True); a validator crash is also skipped — the guard never
    blocks delivery, it only catches provably-broken output."""
    f = (fmt or "").strip().lower().lstrip(".")
    try:
        if not data:
            return ValidationResult(False, f, "empty", "artifact is empty")
        if f == "pdf":
            return _valid_pdf(data)
        if f in _OPENXML or f == "zip":
            return _valid_zip_container(data, f)
        if f == "7z":
            return _valid_7z(data)
        if f == "json":
            return _valid_json(data)
        if f == "csv":
            return _valid_csv(data)
        if f in ("md", "markdown", "txt"):
            return _valid_text(data, f)
        return ValidationResult(True, f, "skipped", "no validator")
    except Exception as exc:  # noqa: BLE001 — the guard must never block
        return ValidationResult(True, f, "skipped", f"validator error: {exc}")


def render_validated(content: str, fmt: str, *, title: str = "",
                     export_settings=None, language: str = "") -> tuple:
    """Public entry: the closed loop + Phase-6 metrics recording."""
    out = _render_validated_impl(content, fmt, title=title,
                                 export_settings=export_settings,
                                 language=language)
    _record(out[3])
    return out


def _render_validated_impl(content: str, fmt: str, *, title: str = "",
                           export_settings=None, language: str = "") -> tuple:
    """render → validate → repair (one re-render) → degrade.

    Returns (data, mime, ext, meta) where meta records what happened:
    {"validated": bool, "method": ..., "repaired": bool,
     "degraded_from": fmt|None, "reason": ...}. Flag-gated by
    `cfg.artifact_validation`; disabled → exact legacy behavior (single
    render, no validation).
    """
    from app.documents.generators import render_document

    try:
        from app.core.config_loader import cfg
        av = cfg.artifact_validation
        enabled = bool(getattr(av, "enabled", True))
        do_repair = bool(getattr(av, "repair", True))
        do_degrade = bool(getattr(av, "degrade", True))
    except Exception:  # noqa: BLE001
        enabled, do_repair, do_degrade = True, True, True

    data, mime, ext = render_document(content, fmt, title=title,
                                      export_settings=export_settings,
                                      language=language)
    meta = {"validated": False, "method": "disabled", "repaired": False,
            "degraded_from": None, "reason": ""}
    if not enabled:
        return data, mime, ext, meta

    v = validate_artifact(data, fmt)
    meta.update({"validated": v.ok, "method": v.method, "reason": v.reason})
    if v.ok:
        return data, mime, ext, meta

    # Repair: one clean re-render — transient generator state (font cache,
    # stream position bugs) is the common cause of a bad first render.
    if do_repair:
        try:
            data2, mime2, ext2 = render_document(content, fmt, title=title, export_settings=export_settings, language=language)
            v2 = validate_artifact(data2, fmt)
            if v2.ok:
                meta.update({"validated": True, "method": v2.method,
                             "repaired": True, "reason": ""})
                log.info("artifact repair succeeded for %s", fmt)
                return data2, mime2, ext2, meta
        except Exception as exc:  # noqa: BLE001
            log.info("artifact repair render failed for %s: %s", fmt, exc)

    # Degrade: walk the capability fallback chain (pdf→docx→md, 7z→zip, …)
    # until something validates — a correct artifact in a simpler format
    # beats a corrupt one in the requested format.
    if do_degrade:
        try:
            from app.capabilities.registry import _FORMAT_FALLBACK
            seen = {(fmt or "").lower()}
            alt = _FORMAT_FALLBACK.get((fmt or "").lower())
            while alt and alt not in seen:
                seen.add(alt)
                try:
                    d3, m3, e3 = render_document(content, alt, title=title, export_settings=export_settings, language=language)
                    v3 = validate_artifact(d3, alt)
                    if v3.ok:
                        meta.update({
                            "validated": True, "method": v3.method,
                            "degraded_from": fmt,
                            "reason": f"{fmt} failed validation ({v.reason}); "
                                      f"delivered {alt} instead"})
                        log.warning("artifact degraded %s -> %s (%s)",
                                    fmt, alt, v.reason)
                        return d3, m3, e3, meta
                except Exception:  # noqa: BLE001
                    pass
                alt = _FORMAT_FALLBACK.get(alt)
        except Exception:  # noqa: BLE001
            pass

    # Nothing validated — ship the original bytes with the failure recorded
    # (delivery is never blocked by its own guard) so the caller can surface it.
    meta.update({"validated": False, "reason": v.reason or "validation failed"})
    return data, mime, ext, meta


def _record(meta: dict) -> None:
    """Phase-6 metrics hook (fail-open)."""
    try:
        from app.obs.decision_metrics import record_artifact_validation
        record_artifact_validation(meta)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["ValidationResult", "validate_artifact", "render_validated"]
