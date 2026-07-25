"""Mermaid backend render lane (vNext §5.4, Stage 8 Component B).

A ```mermaid fence rendered client-side trims to whatever viewport the client
happens to have — long labels clip, wide graphs overflow, and a syntax slip shows
the raw source. §5.4 moves rendering server-side into a real lane:

  1. **lint + normalize** the source — ensure a diagram header + direction, WRAP
     long labels (so text never clips), QUOTE labels with special characters,
     and CAP a runaway graph — all deterministic, no model;
  2. **render** to SVG via `mmdc` (mermaid-cli + Chromium) — an INJECTED seam
     (the binary is on-pod), CACHED by source hash so an unchanged diagram is
     free on re-render;
  3. on a render error, **one fast-tier repair** — hand the mmdc error back to an
     INJECTED repair function once, then re-render;
  4. emit the envelope `{mermaid_source, svg_artifact_id}` (a MEASURED SVG — it
     never trims).

`diagram_gate.should_diagram` already decides WHETHER to draw; this owns the HOW.
The lint/normalize/cache logic is pure + unit-tested on the dev box; the mmdc
render + LLM repair are injected. Fail-open: disabled OR any error → the original
source with no SVG (the client falls back to today's fence). Flag-gated
(`response_arch.mermaid_lane`, default OFF).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Recognized mermaid headers → the canonical directive that must lead the source.
_HEADERS = ("graph", "flowchart", "sequencediagram", "classdiagram",
            "statediagram", "erdiagram", "gantt", "pie", "journey",
            "mindmap", "gitgraph", "timeline")
# Diagrams whose layout takes a direction token (TD/LR); others must NOT get one.
_DIRECTIONAL = ("graph", "flowchart")
_NODE_LABEL = re.compile(r"\[([^\]\n]+)\]|\(([^)\n]+)\)|\{([^}\n]+)\}")
_SPECIAL = re.compile(r"[()#:;<>&\"]")


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.response_arch, "mermaid_lane", False))
    except Exception:  # noqa: BLE001
        return False


def _cfg_int(name: str, default: int) -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.response_arch, name, default) or default)
    except Exception:  # noqa: BLE001
        return default


def strip_fence(text: str) -> str:
    """Return the mermaid body from a ```mermaid fenced block (or the text as-is
    if it isn't fenced). Never raises."""
    try:
        m = re.search(r"```mermaid\b\s*\n(.*?)```", text or "", re.S | re.I)
        return (m.group(1) if m else (text or "")).strip()
    except Exception:  # noqa: BLE001
        return (text or "").strip()


def source_hash(source: str) -> str:
    """Stable cache key for a normalized source."""
    return hashlib.sha256((source or "").encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class NormalizedDiagram:
    source: str
    header: str                        # e.g. "flowchart"
    warnings: list[str] = field(default_factory=list)
    changed: bool = False
    node_count: int = 0
    capped: bool = False

    def to_dict(self) -> dict:
        return {"header": self.header, "warnings": list(self.warnings),
                "changed": self.changed, "node_count": self.node_count,
                "capped": self.capped}


def _detect_header(first_line: str) -> str:
    low = first_line.strip().lower().replace(" ", "")
    for h in _HEADERS:
        if low.startswith(h):
            return h
    return ""


def _wrap_label(label: str, width: int) -> str:
    """Insert <br/> so no unbroken run exceeds `width` — mermaid honours <br/> in
    labels, so text wraps instead of clipping. Preserves existing breaks."""
    if "<br" in label or len(label) <= width:
        return label
    words = label.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br/>".join(lines)


def lint_and_normalize(source: str, *, max_nodes: int | None = None,
                       wrap: int | None = None) -> NormalizedDiagram:
    """Deterministically clean a mermaid source: ensure a header (+ direction for
    flow graphs), wrap long node labels, quote labels with special characters,
    and cap a runaway graph. Never raises → the original on error."""
    try:
        raw = strip_fence(source)
        if not raw:
            return NormalizedDiagram("", "", ["empty source"], False, 0, False)
        cap = max_nodes if max_nodes is not None else _cfg_int("mermaid_max_nodes", 60)
        width = wrap if wrap is not None else _cfg_int("mermaid_label_wrap", 24)
        lines = raw.splitlines()
        warnings: list[str] = []
        changed = False

        # 1) Header + direction.
        header = _detect_header(lines[0]) if lines else ""
        if not header:
            header = "flowchart"
            lines.insert(0, "flowchart TD")
            warnings.append("no diagram header — assumed flowchart TD")
            changed = True
        elif header in _DIRECTIONAL:
            # Ensure a direction token follows a bare "graph"/"flowchart".
            if not re.search(r"\b(TD|TB|BT|LR|RL)\b", lines[0], re.I):
                lines[0] = lines[0].rstrip() + " TD"
                warnings.append("added default direction TD")
                changed = True

        # 2) Per-line: wrap + quote node labels.
        node_count = 0
        for i, line in enumerate(lines):
            def _fix(m: "re.Match") -> str:
                nonlocal changed
                inner = next(g for g in m.groups() if g is not None)
                open_ch = m.group(0)[0]
                close_ch = m.group(0)[-1]
                new = _wrap_label(inner.strip(), width)
                # Quote a label containing specials so mermaid doesn't choke.
                if _SPECIAL.search(new) and not (new.startswith('"') and new.endswith('"')):
                    new = '"' + new.replace('"', "'") + '"'
                if new != inner:
                    changed = True
                return f"{open_ch}{new}{close_ch}"
            new_line, n = _NODE_LABEL.subn(_fix, line)
            node_count += n
            lines[i] = new_line

        # 3) Cap a runaway graph.
        capped = False
        if node_count > cap:
            warnings.append(f"diagram has {node_count} nodes (> cap {cap}) — "
                            "consider splitting; rendering may be dense")
            capped = True

        return NormalizedDiagram("\n".join(lines).strip(), header, warnings,
                                 changed, node_count, capped)
    except Exception:  # noqa: BLE001
        return NormalizedDiagram(strip_fence(source), "", ["normalize error"],
                                 False, 0, False)


@dataclass
class MermaidResult:
    ok: bool
    source: str                        # the normalized source
    svg: str = ""                      # the rendered SVG (empty if not rendered)
    artifact_hash: str = ""            # cache key / svg_artifact_id seed
    header: str = ""
    warnings: list[str] = field(default_factory=list)
    repaired: bool = False
    from_cache: bool = False
    error: str = ""

    def envelope(self) -> dict:
        """The §5.4 envelope shape."""
        return {"mermaid_source": self.source,
                "svg_artifact_id": self.artifact_hash if self.svg else None,
                "warnings": list(self.warnings), "repaired": self.repaired}


async def render(source: str, *, render_fn=None, repair_fn=None,
                 cache: dict | None = None) -> MermaidResult:
    """Normalize → (cache) → mmdc render → one repair on error → envelope.

    `render_fn(mermaid_source) -> svg_str` and `repair_fn(source, error) ->
    fixed_source` are INJECTED (mmdc + a fast-tier LLM on-pod). `cache` maps
    source_hash → svg. Fail-open: disabled / no render_fn / any error → an ok
    result carrying the normalized source but NO svg (client falls back to the
    fence). Never raises."""
    norm = lint_and_normalize(source)
    if not norm.source:
        return MermaidResult(False, "", error="empty source")
    base = MermaidResult(True, norm.source, header=norm.header,
                         warnings=list(norm.warnings),
                         artifact_hash=source_hash(norm.source))
    if not enabled() or render_fn is None:
        return base            # normalized but unrendered — the fail-open path
    try:
        h = base.artifact_hash
        if cache is not None and h in cache:
            base.svg, base.from_cache = cache[h], True
            return base
        try:
            svg = await render_fn(norm.source)
        except Exception as exc:  # noqa: BLE001 — try one repair
            if repair_fn is None:
                base.error = f"render failed: {exc}"
                return base
            try:
                fixed = await repair_fn(norm.source, str(exc))
                fixed_norm = lint_and_normalize(fixed)
                svg = await render_fn(fixed_norm.source)
                base.source = fixed_norm.source
                base.artifact_hash = source_hash(fixed_norm.source)
                base.repaired = True
                base.warnings = list(fixed_norm.warnings)
            except Exception as exc2:  # noqa: BLE001
                base.error = f"repair failed: {exc2}"
                return base
        base.svg = svg or ""
        if base.svg and cache is not None:
            cache[base.artifact_hash] = base.svg
        return base
    except Exception as exc:  # noqa: BLE001
        base.error = f"error: {exc}"
        return base


__all__ = ["enabled", "strip_fence", "source_hash", "NormalizedDiagram",
           "lint_and_normalize", "MermaidResult", "render"]
