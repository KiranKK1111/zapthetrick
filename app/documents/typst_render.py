"""Typst-backed PDF rendering (vNext §3.3, image-bake slice).

Typst is a single static binary that compiles a markup document to a PDF in
~100 ms with far better typography than the office-lib path. This module:

  1. converts our Markdown to a minimal, SAFE Typst document (headings,
     paragraphs, bullet/numbered lists, bold/italic, inline + fenced code), and
  2. compiles it with the `typst` binary in a temp dir → PDF bytes.

**Infra note.** The binary is baked into the pod image (§6.1); a dev box without
it simply reports `available() == False`, so `render_pdf` returns ``None`` and
the caller falls back to the existing fpdf2/weasyprint renderer. Nothing here is
load-bearing — Typst is a typography UPGRADE, never a dependency. Flag-gated
(`documents.typst`, default OFF) so output is byte-identical until switched on
AND the binary is present. Fail-open throughout (never raises).

The Markdown→Typst conversion is a deterministic, testable subset; anything it
can't model degrades to escaped prose, and a Typst compile error → ``None`` →
fallback, so a bad conversion can never ship a broken PDF.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "typst", False))
    except Exception:  # noqa: BLE001
        return False


def _binary() -> str | None:
    """Path to the `typst` binary — a configured override, else `$PATH`."""
    try:
        from app.core.config_loader import cfg
        cfgd = (getattr(cfg.documents, "typst_bin", "") or "").strip()
        if cfgd and Path(cfgd).exists():
            return cfgd
    except Exception:  # noqa: BLE001
        pass
    return shutil.which("typst")


def available() -> bool:
    """Whether the Typst binary is present (baked pod image / local install)."""
    return _binary() is not None


# --------------------------------------------------------------------------- #
# Markdown → Typst
# --------------------------------------------------------------------------- #
# Characters Typst treats as markup in prose; escaped with a backslash so a
# literal `#total` or `*note` renders verbatim instead of triggering markup.
_ESC = re.compile(r"([#*_`$\\<>@\[\]])")


def _esc(text: str) -> str:
    return _ESC.sub(r"\\\1", text or "")


# Inline spans applied AFTER escaping, so `**x**` → `*x*` (Typst bold) survives.
_BOLD = re.compile(r"\\\*\\\*(.+?)\\\*\\\*")          # escaped **x**
_ITALIC = re.compile(r"(?<!\\\*)\\\*(?!\\\*)(.+?)\\\*")  # escaped *x* (not **)
_CODE = re.compile(r"\\`(.+?)\\`")                    # escaped `x`


def _inline(text: str) -> str:
    """Escape prose, then re-enable the common inline spans as Typst markup."""
    s = _esc(text)
    s = _BOLD.sub(r"*\1*", s)
    s = _ITALIC.sub(r"_\1_", s)
    s = _CODE.sub(r"`\1`", s)
    return s


def markdown_to_typst(md: str, *, title: str = "") -> str:
    """Convert a Markdown document to a minimal Typst source string. Deterministic
    and side-effect-free; handles headings, paragraphs, bullet/numbered lists,
    fenced code blocks, and inline bold/italic/code. Unknown constructs degrade
    to escaped prose."""
    lines = (md or "").replace("\r\n", "\n").split("\n")
    out: list[str] = [
        '#set page(margin: 2.2cm)',
        '#set text(font: "New Computer Modern", size: 11pt)',
        '#set par(justify: true, leading: 0.65em)',
        '',
    ]
    if title.strip():
        out.append(f"#align(center)[#text(size: 20pt, weight: \"bold\")"
                   f"[{_inline(title.strip())}]]")
        out.append("")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block — emit as a Typst raw block (verbatim, no escaping).
        m = re.match(r"^```([\w+#.\-]*)\s*$", stripped)
        if m:
            lang = m.group(1)
            body: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # consume the closing fence
            out.append(f"```{lang}")
            out.extend(body)
            out.append("```")
            out.append("")
            continue

        # Heading — `#`..`######` → Typst `=`..`======`.
        hm = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if hm:
            out.append("=" * len(hm.group(1)) + " " + _inline(hm.group(2)))
            out.append("")
            i += 1
            continue

        # Bullet list item.
        bm = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bm:
            out.append("- " + _inline(bm.group(1)))
            i += 1
            continue

        # Numbered list item.
        nm = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if nm:
            out.append("+ " + _inline(nm.group(1)))
            i += 1
            continue

        # Blank line → paragraph break.
        if not stripped:
            out.append("")
            i += 1
            continue

        # Plain paragraph line.
        out.append(_inline(stripped))
        i += 1

    return "\n".join(out).strip() + "\n"


def render_pdf(md: str, *, title: str = "", timeout: float = 20.0) -> bytes | None:
    """Compile the Markdown to a PDF with the Typst binary. Returns the PDF bytes,
    or ``None`` on ANY failure (binary absent, compile error, timeout, empty
    output) so the caller falls back to the existing renderer. Never raises."""
    if not (md or "").strip():
        return None
    binary = _binary()
    if binary is None:
        return None
    try:
        src = markdown_to_typst(md, title=title)
    except Exception:  # noqa: BLE001
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="dtt-typst-") as tmp:
            tdir = Path(tmp)
            typ = tdir / "doc.typ"
            pdf = tdir / "doc.pdf"
            typ.write_text(src, encoding="utf-8")
            proc = subprocess.run(
                [binary, "compile", str(typ), str(pdf)],
                capture_output=True, timeout=timeout, cwd=str(tdir))
            if proc.returncode != 0 or not pdf.exists():
                log.info("typst compile failed (rc=%s): %s",
                         proc.returncode, (proc.stderr or b"")[:200])
                return None
            data = pdf.read_bytes()
            return data or None
    except Exception as exc:  # noqa: BLE001
        log.info("typst render error: %s", exc)
        return None


__all__ = ["enabled", "available", "markdown_to_typst", "render_pdf"]
