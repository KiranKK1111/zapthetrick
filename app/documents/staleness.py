"""Multi-format staleness — Phase 5 of the Document Generation roadmap.

DocuementGeneration.md's "Multi-format Synchronization": when the source document
gets a new version, the OTHER formats already exported from it (the PDF, the
DOCX, the slides) are now out of date and should be regenerated. This module is
the deterministic staleness logic over the Phase-5 version store.

A ``doc_key`` names one logical document; each stored version records the
``doc_format`` it was produced in. A format is **stale** when the newest version
carrying that format is older than the document's newest version overall — i.e.
the source moved on and that format wasn't re-rendered from it.

Pure + fail-open: ``compute_staleness`` takes plain ``(version, format)`` pairs
so it's trivially testable; the endpoint wraps it around the store.
"""
from __future__ import annotations


def compute_staleness(versions) -> dict:
    """Given the version rows of one document (each an object with ``.version`` +
    ``.doc_format``, or a ``(version, format)`` / ``{"version","format"}`` pair),
    return a staleness report:

        {"latest_version": N,
         "formats": [{"format": f, "last_version": v, "stale": bool}, ...],
         "stale_formats": [f, ...],
         "any_stale": bool}

    A format is stale when the newest version that produced it is older than the
    document's newest version. Empty input → an empty, non-stale report."""
    parsed: list[tuple[int, str]] = []
    for v in versions or []:
        if isinstance(v, dict):
            ver = v.get("version")
            fmt = v.get("format") or v.get("doc_format")
        elif isinstance(v, (tuple, list)) and len(v) >= 2:
            ver, fmt = v[0], v[1]
        else:
            ver = getattr(v, "version", None)
            fmt = getattr(v, "doc_format", None) or getattr(v, "format", None)
        try:
            ver = int(ver)
        except (TypeError, ValueError):
            continue
        fmt = (str(fmt or "").strip().lower()) or "?"
        parsed.append((ver, fmt))

    if not parsed:
        return {"latest_version": 0, "formats": [], "stale_formats": [],
                "any_stale": False}

    latest = max(v for v, _ in parsed)
    # Newest version each format was produced at.
    last_for_fmt: dict[str, int] = {}
    for ver, fmt in parsed:
        last_for_fmt[fmt] = max(last_for_fmt.get(fmt, 0), ver)

    formats = []
    stale_formats = []
    for fmt in sorted(last_for_fmt):
        last_v = last_for_fmt[fmt]
        stale = last_v < latest
        formats.append({"format": fmt, "last_version": last_v, "stale": stale})
        if stale:
            stale_formats.append(fmt)
    return {"latest_version": latest, "formats": formats,
            "stale_formats": stale_formats, "any_stale": bool(stale_formats)}


async def staleness_for_document(session, doc_key) -> dict:
    """Load a document's versions from the store and compute its multi-format
    staleness. Fail-open to an empty report."""
    try:
        from app.documents.store import list_versions
        rows = await list_versions(session, doc_key)
    except Exception:  # noqa: BLE001
        rows = []
    report = compute_staleness(rows)
    report["doc_key"] = str(doc_key)
    return report


__all__ = ["compute_staleness", "staleness_for_document"]
