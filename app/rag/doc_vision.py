"""Layout-aware + chart understanding for documents (roadmap Phase 3 #21).

The parser (`documents/parser.py`) extracts text + tables and hands images to a
generic vision model. The audit's gap: "Text + tables only; no layout/chart
understanding." This module adds a real, minimal, dependency-free layer on top of
the markdown the parser already produces:

  * LAYOUT — the document's structural skeleton: heading hierarchy (H1..H6),
    section count, table/figure/code-block inventory, and a max heading depth,
    so a consumer knows the shape of what it ingested (a flat wall of text vs a
    structured spec vs a slide deck).

  * CHART UNDERSTANDING — a Markdown/GFM table is the textual form of a chart.
    For each table we identify the numeric columns and summarize them (min, max,
    monotonic trend up/down/flat, and which label row holds the extreme), i.e.
    we *understand* the data the table plots instead of storing it as opaque
    text. This is the "beyond generic vision" minimum: a caption a downstream
    turn can cite ("Revenue peaks in Q4 at 120; trend is upward").

Wired into `rag/documents.py::ingest_chat_document`: each ingested document gets
a compact `layout` summary the turn can reference. Deterministic, offline,
fail-open — a malformed table degrades to "no charts", never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:]*-{2,}[-\s|:]*\|?\s*$")
_CODE_FENCE = re.compile(r"^\s*```")
_FIGURE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b", re.IGNORECASE)
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


@dataclass
class ChartInsight:
    title: str
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "columns": self.columns, "rows": self.rows,
                "numeric_columns": self.numeric_columns, "summary": self.summary}


@dataclass
class Layout:
    headings: list[dict] = field(default_factory=list)   # [{level,text}]
    max_depth: int = 0
    sections: int = 0
    tables: int = 0
    figures: int = 0
    code_blocks: int = 0
    charts: list[ChartInsight] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "max_depth": self.max_depth, "sections": self.sections,
            "tables": self.tables, "figures": self.figures,
            "code_blocks": self.code_blocks,
            "headings": self.headings[:40],
            "charts": [c.as_dict() for c in self.charts],
        }


def _to_number(cell: str):
    s = (cell or "").strip().replace(",", "").replace("$", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_table(block: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse a Markdown table block into (headers, data_rows)."""
    def cells(line: str) -> list[str]:
        inner = _TABLE_ROW.match(line).group(1)
        return [c.strip() for c in inner.split("|")]

    headers = cells(block[0])
    data = [cells(r) for r in block[2:]]         # skip header + separator row
    return headers, data


def _chart_from_table(headers, data) -> ChartInsight | None:
    if not headers or not data:
        return None
    ncols = len(headers)
    label_col = 0
    numeric_cols: list[int] = []
    for ci in range(ncols):
        vals = [_to_number(r[ci]) for r in data if ci < len(r)]
        if vals and sum(1 for v in vals if v is not None) >= max(2, len(vals) // 2):
            numeric_cols.append(ci)
    if not numeric_cols:
        return None
    # The first non-numeric column is the label axis; else the row index.
    for ci in range(ncols):
        if ci not in numeric_cols:
            label_col = ci
            break

    insight = ChartInsight(
        title=(headers[label_col] if label_col < len(headers) else "table"),
        columns=list(headers), rows=len(data),
        numeric_columns=[headers[ci] for ci in numeric_cols if ci < len(headers)],
    )
    # Summarize the FIRST numeric column (the primary series).
    series_ci = numeric_cols[0]
    pairs = []
    for r in data:
        if series_ci >= len(r):
            continue
        v = _to_number(r[series_ci])
        if v is None:
            continue
        label = r[label_col] if label_col < len(r) else f"row{len(pairs)}"
        pairs.append((label, v))
    if not pairs:
        return insight
    vals = [v for _l, v in pairs]
    hi_label, hi = max(pairs, key=lambda p: p[1])
    lo_label, lo = min(pairs, key=lambda p: p[1])
    trend = "flat"
    if len(vals) >= 2:
        if all(b >= a for a, b in zip(vals, vals[1:])) and vals[-1] > vals[0]:
            trend = "upward"
        elif all(b <= a for a, b in zip(vals, vals[1:])) and vals[-1] < vals[0]:
            trend = "downward"
    col_name = headers[series_ci] if series_ci < len(headers) else "value"
    insight.summary = (
        f"{col_name} ranges {lo:g}–{hi:g}; peaks at '{hi_label}' ({hi:g}), "
        f"lowest at '{lo_label}' ({lo:g}); trend is {trend}.")
    return insight


def analyze_layout(markdown: str) -> Layout:
    """Extract the document's structural + chart layout from its markdown.
    Deterministic + fail-open — never raises."""
    layout = Layout()
    try:
        lines = (markdown or "").splitlines()
        in_code = False
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if _CODE_FENCE.match(line):
                if not in_code:
                    layout.code_blocks += 1
                in_code = not in_code
                i += 1
                continue
            if in_code:
                i += 1
                continue
            m = _HEADING.match(line)
            if m:
                level = len(m.group(1))
                layout.headings.append({"level": level, "text": m.group(2)[:120]})
                layout.max_depth = max(layout.max_depth, level)
                i += 1
                continue
            layout.figures += len(_FIGURE.findall(line))
            # Table block: a row, a separator, then ≥1 data row.
            if (_TABLE_ROW.match(line) and i + 1 < n
                    and _TABLE_SEP.match(lines[i + 1])):
                block = [line, lines[i + 1]]
                j = i + 2
                while j < n and _TABLE_ROW.match(lines[j]):
                    block.append(lines[j])
                    j += 1
                if len(block) >= 3:
                    layout.tables += 1
                    try:
                        headers, data = _parse_table(block)
                        ch = _chart_from_table(headers, data)
                        if ch is not None:
                            layout.charts.append(ch)
                    except Exception:  # noqa: BLE001
                        pass
                i = j
                continue
            i += 1
        layout.sections = len(layout.headings)
    except Exception:  # noqa: BLE001 — layout is additive, never fatal
        return layout
    return layout


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "doc_vision", True))
    except Exception:  # noqa: BLE001
        return True


__all__ = ["Layout", "ChartInsight", "analyze_layout", "enabled"]
