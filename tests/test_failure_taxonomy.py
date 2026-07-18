"""Tests for the formal failure taxonomy (roadmap Phase 1 #5)."""
from __future__ import annotations

import pytest

from app.obs import failure_taxonomy as ft
from app.obs.failure_taxonomy import Recovery, Severity

_RETRY_FAMILY = {Recovery.RETRY, Recovery.RETRY_DIFFERENT, Recovery.COOLDOWN_WAIT,
                 Recovery.REPAIR, Recovery.REPLAN}


def test_ids_unique_and_match_keys():
    for key, fc in ft.TAXONOMY.items():
        assert key == fc.id
    assert len(ft.TAXONOMY) == len({fc.id for fc in ft.all_classes()})


def test_every_class_well_formed():
    for fc in ft.all_classes():
        assert fc.id and fc.title and fc.description
        assert isinstance(fc.severity, Severity)
        assert isinstance(fc.recovery, Recovery)
        assert isinstance(fc.retryable, bool)


def test_retryable_consistency():
    # A retryable failure must carry a recovery from the retry family; a
    # non-retryable one must NOT claim a plain-retry strategy (that would be a
    # blind retry — anti-pattern rule #10).
    for fc in ft.all_classes():
        if fc.retryable:
            assert fc.recovery in _RETRY_FAMILY, (
                f"{fc.id} is retryable but recovery={fc.recovery} isn't a retry strategy."
            )
        else:
            assert fc.recovery is not Recovery.RETRY, (
                f"{fc.id} is not retryable yet recovery=RETRY (blind retry risk)."
            )


def test_core_surface_covered():
    # The taxonomy must cover the platform's real failure surface, not a subset.
    required = {
        "stt_unavailable", "decision_skip", "retrieval_error",
        "provider_rate_limit", "provider_transport", "provider_auth",
        "generation_timeout", "verification_failed", "sandbox_timeout",
        "network_error", "internal_error",
    }
    missing = required - set(ft.TAXONOMY)
    assert not missing, f"Taxonomy missing core failure class(es): {sorted(missing)}"


def test_get_and_unknown():
    assert ft.get("provider_rate_limit").recovery is Recovery.COOLDOWN_WAIT
    assert ft.get("does_not_exist") is None


@pytest.mark.parametrize("exc,expected", [
    (TimeoutError("operation timed out"), "generation_timeout"),
    (RuntimeError("HTTP 429 Too Many Requests"), "provider_rate_limit"),
    (RuntimeError("could not decrypt api key"), "provider_auth"),
    (ConnectionError("connection reset by peer"), "provider_transport"),
    (ValueError("SyntaxError: invalid syntax"), "verification_failed"),
    (RuntimeError("something totally unexpected"), "internal_error"),
])
def test_classify_exception(exc, expected):
    assert ft.classify_exception(exc).id == expected


def test_classify_never_raises():
    class Weird(Exception):
        def __str__(self):
            raise RuntimeError("boom")
    # Must degrade to internal_error, not propagate.
    assert ft.classify_exception(Weird()).id == "internal_error"
