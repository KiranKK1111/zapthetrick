"""Universal preview matrix (vNext §8.8, Stage 8 Component C).

Every artifact — a pdf, a spreadsheet, a zip, a react snippet — deserves a
Claude-style in-app preview, not just a download link. §8.8 routes each format to
the right preview MODE and builds it:

  * **pdf** → rasterize pages natively;
  * **docx / pptx** → headless LibreOffice → pdf → page rasters;
  * **xlsx / csv** → typed JSON per sheet (columns + rows + inferred types);
  * **html / react / svg** → sandboxed CSP render payload;
  * **zip / 7z** → a member TREE (from the §3.3 verify namelist);
  * **txt / md / code / json** → text / JSON tree.

Page-1 is served BEFORE the full document verifies (so the panel fills instantly),
and every preview is cached per `(artifact, version)`. This module owns the
deterministic parts — the format→mode DESCRIPTOR, the pure builders (csv→sheets,
zip→tree, json, text), and the cache key — while the rasterize (LibreOffice /
pdf, on-pod) and the xlsx sheet extraction are INJECTED seams, unit-tested with
stubs on the dev box. Fail-open. Flag-gated (`documents.preview_panel`, OFF).
"""
from __future__ import annotations

import csv as _csv
import io
import json as _json
from dataclasses import dataclass, field

# Preview modes.
RASTER = "raster"            # page images (pdf native / docx/pptx via LO)
SHEETS = "sheets"           # typed JSON per sheet (xlsx/csv)
SANDBOX = "sandbox"         # CSP-sandboxed html/svg/react
TEXT = "text"               # plain text / markdown / source code
JSON_TREE = "json_tree"     # structured json
MEMBER_TREE = "member_tree"  # archive member listing
UNSUPPORTED = "unsupported"

# Format → (mode, needs_rasterize, needs_sandbox). Source-code formats fall
# through to TEXT via the default.
_MATRIX: dict[str, tuple[str, bool, bool]] = {
    "pdf": (RASTER, True, False),
    "docx": (RASTER, True, False),
    "pptx": (RASTER, True, False),
    "xlsx": (SHEETS, False, False),
    "csv": (SHEETS, False, False),
    "html": (SANDBOX, False, True),
    "svg": (SANDBOX, False, True),
    "react": (SANDBOX, False, True),
    "jsx": (SANDBOX, False, True),
    "json": (JSON_TREE, False, False),
    "zip": (MEMBER_TREE, False, False),
    "7z": (MEMBER_TREE, False, False),
    "txt": (TEXT, False, False),
    "md": (TEXT, False, False),
}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "preview_panel", False))
    except Exception:  # noqa: BLE001
        return False


def _norm_fmt(fmt: str) -> str:
    return (fmt or "").strip().lower().lstrip(".")


@dataclass
class PreviewDescriptor:
    fmt: str
    mode: str
    needs_rasterize: bool = False
    needs_sandbox: bool = False
    page_1_first: bool = False     # raster streams page-1 before the full verify
    cacheable: bool = True

    def to_dict(self) -> dict:
        return {"fmt": self.fmt, "mode": self.mode,
                "needs_rasterize": self.needs_rasterize,
                "needs_sandbox": self.needs_sandbox,
                "page_1_first": self.page_1_first, "cacheable": self.cacheable}


def preview_descriptor(fmt: str) -> PreviewDescriptor:
    """Route a format to its preview mode. An unknown format that looks like
    source code → TEXT; anything else → UNSUPPORTED. Never raises."""
    try:
        f = _norm_fmt(fmt)
        if f in _MATRIX:
            mode, raster, sandbox = _MATRIX[f]
            return PreviewDescriptor(f, mode, raster, sandbox,
                                     page_1_first=raster, cacheable=True)
        # Unknown → treat a code-ish extension as text, else unsupported.
        if f and f.isalnum() and len(f) <= 8:
            return PreviewDescriptor(f, TEXT, False, False, False, True)
        return PreviewDescriptor(f, UNSUPPORTED, False, False, False, False)
    except Exception:  # noqa: BLE001
        return PreviewDescriptor(_norm_fmt(fmt), UNSUPPORTED)


def preview_cache_key(artifact_id, version, fmt: str, page: int = 0) -> str:
    """Cache key per (artifact, version, format, page). Stable + collision-safe."""
    return f"{artifact_id}:{version}:{_norm_fmt(fmt)}:{page}"


# --------------------------------------------------------------------------- #
# Pure builders
# --------------------------------------------------------------------------- #
def _infer_type(values: "list[str]") -> str:
    """Column type from its non-empty cells: int | float | bool | str."""
    seen = [v for v in values if (v or "").strip() != ""]
    if not seen:
        return "str"
    def _is_int(v):
        try:
            int(v); return True
        except (ValueError, TypeError):
            return False
    def _is_float(v):
        try:
            float(v); return True
        except (ValueError, TypeError):
            return False
    if all(_is_int(v) for v in seen):
        return "int"
    if all(_is_float(v) for v in seen):
        return "float"
    if all((v or "").strip().lower() in ("true", "false") for v in seen):
        return "bool"
    return "str"


@dataclass
class Sheet:
    name: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "columns": self.columns, "rows": self.rows,
                "types": self.types, "truncated": self.truncated}


def sheets_from_csv(text: str, *, name: str = "Sheet1",
                    max_rows: int = 500) -> "list[Sheet]":
    """Parse CSV text → one typed Sheet. Header row → columns; column types are
    inferred from the data. Never raises → an empty sheet."""
    try:
        reader = list(_csv.reader(io.StringIO(text or "")))
        if not reader:
            return [Sheet(name)]
        header = reader[0]
        body = reader[1:]
        truncated = len(body) > max_rows
        body = body[:max_rows]
        cols = len(header)
        types = [_infer_type([r[i] if i < len(r) else "" for r in body])
                 for i in range(cols)]
        return [Sheet(name=name, columns=header, rows=body, types=types,
                      truncated=truncated)]
    except Exception:  # noqa: BLE001
        return [Sheet(name)]


def member_tree(names: "list[str]") -> dict:
    """Build a nested member tree from a flat archive namelist. Directories are
    dicts, files are None leaves. Never raises."""
    tree: dict = {}
    try:
        for raw in names or ():
            path = (raw or "").strip().replace("\\", "/").lstrip("/")
            if not path:
                continue
            parts = [p for p in path.split("/") if p]
            node = tree
            for i, part in enumerate(parts):
                is_leaf = (i == len(parts) - 1) and not path.endswith("/")
                if is_leaf:
                    node.setdefault(part, None)
                else:
                    nxt = node.get(part)
                    if not isinstance(nxt, dict):
                        nxt = {}
                        node[part] = nxt
                    node = nxt
        return tree
    except Exception:  # noqa: BLE001
        return tree


def member_tree_from_zip(data: bytes) -> dict:
    """Member tree from raw zip bytes (stdlib, no external tool). Fail-open → {}."""
    try:
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return member_tree(zf.namelist())
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class PreviewResult:
    ok: bool
    fmt: str
    mode: str
    payload: dict = field(default_factory=dict)   # mode-specific content
    page_count: int = 0
    cache_key: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "fmt": self.fmt, "mode": self.mode,
                "payload": self.payload, "page_count": self.page_count,
                "cache_key": self.cache_key, "error": self.error}


async def build_preview(fmt: str, *, text: str | None = None,
                        data: bytes | None = None, artifact_id="", version=1,
                        rasterize_fn=None, sheet_fn=None,
                        cache: dict | None = None) -> PreviewResult:
    """Build a preview for `fmt`, dispatching on the descriptor. Pure modes
    (sheets/text/json/member-tree) are computed inline; RASTER uses the INJECTED
    `rasterize_fn(data|text, fmt) -> list[page_image]`, and an xlsx sheet extract
    uses the INJECTED `sheet_fn(data) -> list[Sheet]`. Cached per
    (artifact, version, fmt). Fail-open: disabled / unsupported / any error → a
    not-ok result. Never raises."""
    d = preview_descriptor(fmt)
    key = preview_cache_key(artifact_id, version, d.fmt)
    if not enabled():
        return PreviewResult(False, d.fmt, d.mode, cache_key=key, error="disabled")
    if cache is not None and key in cache:
        return cache[key]
    try:
        res: PreviewResult
        if d.mode == TEXT:
            res = PreviewResult(True, d.fmt, TEXT, {"text": text or ""},
                                1, key)
        elif d.mode == JSON_TREE:
            try:
                parsed = _json.loads(text or "null")
            except Exception:  # noqa: BLE001 — show raw on bad json
                parsed = None
            res = PreviewResult(True, d.fmt, JSON_TREE,
                                {"json": parsed, "raw": text or ""}, 1, key)
        elif d.mode == SANDBOX:
            res = PreviewResult(True, d.fmt, SANDBOX,
                                {"html": text or "", "csp": True}, 1, key)
        elif d.mode == SHEETS:
            if d.fmt == "csv":
                sheets = sheets_from_csv(text or "")
            elif sheet_fn is not None:
                sheets = await sheet_fn(data) if data is not None else []
            else:
                sheets = []          # xlsx needs the injected extractor
            res = PreviewResult(True, d.fmt, SHEETS,
                                {"sheets": [s.to_dict() for s in sheets]},
                                len(sheets), key)
        elif d.mode == MEMBER_TREE:
            tree = member_tree_from_zip(data) if data is not None else {}
            res = PreviewResult(True, d.fmt, MEMBER_TREE, {"tree": tree}, 1, key)
        elif d.mode == RASTER:
            if rasterize_fn is None:
                return PreviewResult(False, d.fmt, RASTER, cache_key=key,
                                     error="no rasterizer")
            pages = await rasterize_fn(data if data is not None else text, d.fmt)
            pages = list(pages or [])
            res = PreviewResult(True, d.fmt, RASTER,
                                {"pages": pages, "page_1_first": d.page_1_first},
                                len(pages), key)
        else:
            return PreviewResult(False, d.fmt, UNSUPPORTED, cache_key=key,
                                 error="unsupported format")
        if cache is not None and res.ok:
            cache[key] = res
        return res
    except Exception as exc:  # noqa: BLE001
        return PreviewResult(False, d.fmt, d.mode, cache_key=key,
                             error=f"error: {exc}")


__all__ = ["RASTER", "SHEETS", "SANDBOX", "TEXT", "JSON_TREE", "MEMBER_TREE",
           "UNSUPPORTED", "enabled", "PreviewDescriptor", "preview_descriptor",
           "preview_cache_key", "Sheet", "sheets_from_csv", "member_tree",
           "member_tree_from_zip", "PreviewResult", "build_preview"]
