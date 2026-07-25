"""Provider import validation gate (vNext §2.8).

Pins the pure decision logic: which statuses are usable/quarantined, and the
blip-safe mapping from an at-upload validation result to the stored status.
"""
from __future__ import annotations

from app.llm import import_gate as G


def test_usable_statuses_match_the_router_filter():
    # MUST equal the router's snapshot filter literal (app/llm/router.py:
    # status.in_(("healthy", "unknown"))). A drift here would let a quarantined
    # key back into routing (or exclude a usable one).
    assert G.USABLE_STATUSES == ("healthy", "unknown")


def test_gate_usable_statuses_fill_catalog():
    for s in ("healthy", "unknown"):
        d = G.gate(s)
        assert d["usable"] is True
        assert d["fill_catalog"] is True
        assert d["quarantined"] is False


def test_gate_bad_statuses_are_quarantined():
    for s in ("invalid", "error", "weird"):
        d = G.gate(s)
        assert d["usable"] is False
        assert d["fill_catalog"] is False
        assert d["quarantined"] is True


def test_resolve_confirmed_good_and_bad_pass_through():
    assert G.resolve_upload_status("healthy") == "healthy"
    assert G.resolve_upload_status("invalid") == "invalid"


def test_resolve_inconclusive_error_keeps_key_usable():
    # A 403/transport blip ('error') must NOT sideline a possibly-good key.
    assert G.resolve_upload_status("error", prior="unknown") == "unknown"
    assert G.resolve_upload_status("error", prior="healthy") == "healthy"
    assert G.resolve_upload_status(None) == "unknown"
    # A prior that itself isn't usable collapses to 'unknown', not back to bad.
    assert G.resolve_upload_status("error", prior="invalid") == "unknown"


def test_resolve_is_case_insensitive():
    assert G.resolve_upload_status("HEALTHY") == "healthy"
    assert G.resolve_upload_status("Invalid") == "invalid"


def test_validation_enabled_defaults_on():
    # No `providers` config section → validation on by default (the §2.8 intent).
    assert G.validation_enabled() is True


def test_keys_repo_exposes_set_status():
    from app.llm import keys as keys_repo
    assert hasattr(keys_repo, "set_status")
