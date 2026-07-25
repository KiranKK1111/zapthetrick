"""Provider import validation gate (vNext §2.8).

When a provider key is added it is validated immediately and a confirmed-bad key
is QUARANTINED — stored and visible in the UI, but excluded from routing so it
can never surface as a runtime "no route" failure — instead of sitting untested
until the ~5-minute health sweep notices. Only a usable key's provider gets a
catalog fetch (valid providers → discovery).

Pure + deterministic so the decision logic is unit-testable without the DB /
network. The endpoint (`routes_providers.add_key`) supplies the live validation
and DB writes; this module owns the *decisions*.
"""
from __future__ import annotations

# Single source of truth for the statuses the router treats as USABLE. MUST stay
# in sync with the router's snapshot filter (app/llm/router.py — the
# `status.in_(("healthy", "unknown"))` clause) and the providers routes. A key in
# any other status ("invalid", "error") is quarantined: excluded from routing.
USABLE_STATUSES: tuple[str, ...] = ("healthy", "unknown")


def validation_enabled() -> bool:
    """Whether to validate a key at upload. Default ON (the §2.8 intent);
    disable via `providers.import_validation: false`. Read defensively so the
    absence of a `providers` config section is fine."""
    try:
        from app.core.config_loader import cfg
        prov = getattr(cfg, "providers", None)
        if prov is None:
            return True
        v = getattr(prov, "import_validation", True)
        return True if v is None else bool(v)
    except Exception:  # noqa: BLE001 — never let a config read gate the add
        return True


def resolve_upload_status(validated: str | None, prior: str = "unknown") -> str:
    """Map a fresh key-validation result to the status stored AT UPLOAD.

    * ``healthy`` → confirmed good.
    * ``invalid`` → confirmed bad (401) → quarantine NOW.
    * anything else (``error`` from a 403 / transport blip, or empty) →
      INCONCLUSIVE: keep the key usable (``unknown``) so a momentary network
      hiccup at upload can't sideline a possibly-good key — the health loop
      re-checks it. Mirrors the health loop's "a blip must not drop a working
      key" rule.
    """
    s = (validated or "").strip().lower()
    if s == "healthy":
        return "healthy"
    if s == "invalid":
        return "invalid"
    return prior if prior in USABLE_STATUSES else "unknown"


def gate(status: str) -> dict:
    """The import decision for a stored key's status: is it usable, is it
    quarantined, and should its provider get a catalog fetch (seed + discovery)?
    Only usable keys fill the catalog."""
    usable = status in USABLE_STATUSES
    return {"status": status, "usable": usable,
            "quarantined": not usable, "fill_catalog": usable}


__all__ = ["USABLE_STATUSES", "validation_enabled", "resolve_upload_status",
           "gate"]
