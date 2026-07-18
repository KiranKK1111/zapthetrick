"""Response Fingerprint (roadmap Phase 6 #22 — Deterministic Rendering +
Response Fingerprint).

A stable, content-derived id for a response plus the provenance that produced it
(model, planner/app version, knowledge sources, verification state). Two
identical responses from the same inputs get the same fingerprint — the basis
for reproducibility, dedup, and "did this actually change?" checks. Attached to
the envelope's `meta`. Deterministic + fail-open.
"""
from __future__ import annotations

import hashlib
import json


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def content_hash(text: str) -> str:
    """Stable short hash of answer content (whitespace-normalized)."""
    try:
        norm = " ".join((text or "").split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


def response_fingerprint(
    *,
    content: str = "",
    model: str | None = None,
    app_version: str | None = None,
    knowledge_sources: list[str] | None = None,
    verified: bool | None = None,
) -> dict:
    """Build the fingerprint record. `hash` covers content + provenance, so a
    change in the model or sources yields a new fingerprint even for identical
    text (important for 'why did this answer change?')."""
    try:
        sources = sorted(set(knowledge_sources or []))
        provenance = {
            "content": content_hash(content),
            "model": model or "",
            "app_version": app_version or "",
            "sources": sources,
            "verified": verified,
        }
        digest = hashlib.sha256(_canonical(provenance).encode("utf-8")).hexdigest()[:16]
        fp = {"hash": digest, "content_hash": provenance["content"]}
        if model:
            fp["model"] = model
        if app_version:
            fp["app_version"] = app_version
        if sources:
            fp["sources"] = sources
        if verified is not None:
            fp["verified"] = verified
        return fp
    except Exception:  # noqa: BLE001 — a fingerprint must never break a turn
        return {}


__all__ = ["content_hash", "response_fingerprint"]
