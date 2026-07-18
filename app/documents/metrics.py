"""Document-generation metrics — Phase 8 (subset) / roadmap cross-cutting.

The document's closing advice: let real usage data guide the next improvements.
This is the lightweight, in-memory instrumentation for that — how many documents
were exported, by format, how many failed, and the average render latency, PLUS
the roadmap's three named signals:

  * **regeneration rate** — what fraction of exports were an UPDATE/regeneration
    of an existing document rather than a fresh one (`record_regeneration`).
  * **most-edited sections** — which section headings get rewritten the most
    (`record_section_edit`) — the doc's "most-edited sections" signal.
  * **template success** — per design-template success/fail on export
    (`record_template`) — the doc's "template success" signal.

No persistence (a single-process, single-user counter); read via
`GET /api/documents/metrics`, reset for tests.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()


def _blank() -> dict:
    return {
        "exports": 0,
        "failures": 0,
        "by_format": {},          # fmt -> count
        "total_latency_ms": 0.0,
        "latency_samples": 0,     # exports that actually reported a latency
        "regenerations": 0,       # exports/updates that regenerated an existing doc
        "section_edits": {},      # section heading -> times rewritten
        "templates": {},          # template id -> {"ok": n, "fail": n}
    }


_stats = _blank()


def record_export(fmt: str, *, ok: bool = True, latency_ms: float = 0.0) -> None:
    """Record one export outcome. Never raises.

    ``latency_ms`` is the wall time the render actually took; callers on the
    export path MUST pass it (an export recorded without one is counted, but is
    not allowed to drag the average toward zero — see :func:`snapshot`)."""
    try:
        with _lock:
            _stats["exports"] += 1
            if not ok:
                _stats["failures"] += 1
            f = (fmt or "?").lower()
            _stats["by_format"][f] = _stats["by_format"].get(f, 0) + 1
            if latency_ms and latency_ms > 0:
                _stats["total_latency_ms"] += float(latency_ms)
                _stats["latency_samples"] += 1
    except Exception:  # noqa: BLE001
        pass


def record_regeneration() -> None:
    """One existing document was regenerated / incrementally updated (the
    UPDATE_EXISTING path). Feeds ``regeneration_rate``. Never raises."""
    try:
        with _lock:
            _stats["regenerations"] += 1
    except Exception:  # noqa: BLE001
        pass


def record_section_edit(heading: str) -> None:
    """One section (``heading``) was rewritten by an incremental update. Feeds
    ``most_edited_sections``. A blank heading is ignored. Never raises."""
    try:
        h = (heading or "").strip()
        if not h:
            return
        with _lock:
            _stats["section_edits"][h] = _stats["section_edits"].get(h, 0) + 1
    except Exception:  # noqa: BLE001
        pass


def record_template(name: str, *, ok: bool = True) -> None:
    """One export rendered through design template ``name`` (success or failure).
    Feeds ``template_success``. Never raises."""
    try:
        n = (name or "").strip().lower()
        if not n:
            return
        with _lock:
            slot = _stats["templates"].setdefault(n, {"ok": 0, "fail": 0})
            slot["ok" if ok else "fail"] += 1
    except Exception:  # noqa: BLE001
        pass


def _top_sections(edits: dict, k: int = 5) -> list:
    return [{"section": h, "edits": n}
            for h, n in sorted(edits.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _template_success(templates: dict) -> dict:
    out: dict = {}
    for name, slot in templates.items():
        total = slot["ok"] + slot["fail"]
        out[name] = {
            "ok": slot["ok"], "fail": slot["fail"],
            "success_rate": round(slot["ok"] / total, 3) if total else 0.0,
        }
    return out


def snapshot() -> dict:
    with _lock:
        n = _stats["exports"]
        samples = _stats["latency_samples"]
        regen = _stats["regenerations"]
        return {
            "exports": n,
            "failures": _stats["failures"],
            "by_format": dict(_stats["by_format"]),
            # Averaged over the exports that TIMED themselves, so a caller that
            # records without a latency doesn't silently halve the mean.
            "avg_latency_ms": (round(_stats["total_latency_ms"] / samples, 1)
                               if samples else 0.0),
            "regenerations": regen,
            # Fraction of all exports that were regenerations of an existing doc
            # (the doc's "regeneration rate"). Denominator is exports+regens so a
            # regeneration that never re-exports still counts.
            "regeneration_rate": (round(regen / (n + regen), 3)
                                  if (n + regen) else 0.0),
            "most_edited_sections": _top_sections(_stats["section_edits"]),
            "template_success": _template_success(_stats["templates"]),
        }


def reset() -> None:
    with _lock:
        _stats.clear()
        _stats.update(_blank())


__all__ = ["record_export", "record_regeneration", "record_section_edit",
           "record_template", "snapshot", "reset"]
