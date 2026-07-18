"""
Developer overlay meta (roadmap Phase 2 #29 / 2D-29).

Assembles a compact diagnostic overlay — attributed speaker role, the model that
answered, first-token / total latency, phase and seniority band — into a single
`meta.dev` frame the FE can render as a dev HUD. Purely additive: legacy/prod
clients ignore the unknown field. Deterministic + fail-open.
"""
from __future__ import annotations


def overlay(
    *,
    role: str | None = None,
    model: str | None = None,
    latency_ms: float | None = None,
    first_token_ms: float | None = None,
    phase: str | None = None,
    band: str | None = None,
    qtype: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build the dev-overlay payload (values omitted when unknown). Never
    raises → a minimal dict."""
    try:
        out: dict = {}
        if role:
            out["role"] = role
        if model:
            out["model"] = model
        if latency_ms is not None:
            out["latency_ms"] = int(latency_ms)
        if first_token_ms is not None:
            out["first_token_ms"] = int(first_token_ms)
        if phase:
            out["phase"] = phase
        if band:
            out["band"] = band
        if qtype:
            out["qtype"] = qtype
        if extra:
            for k, v in extra.items():
                out.setdefault(k, v)
        return out
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["overlay"]
