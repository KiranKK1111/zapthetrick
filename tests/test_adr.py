"""Tests for the ADR engine (roadmap Phase 8A #11)."""
from __future__ import annotations

import pytest

from app.core import adr


@pytest.fixture(autouse=True)
def _clean():
    adr.reset()  # restores the seeded set
    yield
    adr.reset()


def test_seeded_with_real_decisions():
    ids = {a.id for a in adr.all_adrs()}
    assert {"ADR-0001", "ADR-0002", "ADR-0003", "ADR-0004"} <= ids
    a1 = adr.get("ADR-0001")
    assert a1.status == adr.ACCEPTED and a1.rationale


def test_record_and_query():
    adr.record(adr.ADR("ADR-0100", "Use X", rationale="because Y"))
    got = adr.get("ADR-0100")
    assert got.title == "Use X" and got.status == adr.ACCEPTED


def test_supersession_chain():
    adr.record(adr.ADR("ADR-0200", "Old approach", rationale="v1"))
    adr.record(adr.ADR("ADR-0201", "New approach", rationale="v2", supersedes="ADR-0200"))
    old = adr.get("ADR-0200")
    new = adr.get("ADR-0201")
    assert old.status == adr.SUPERSEDED and old.superseded_by == "ADR-0201"
    assert new.status == adr.ACCEPTED
    assert "ADR-0200" not in {a.id for a in adr.active()}


def test_deprecate():
    adr.record(adr.ADR("ADR-0300", "Temp", rationale="z"))
    adr.deprecate("ADR-0300")
    assert adr.get("ADR-0300").status == adr.DEPRECATED
    assert "ADR-0300" not in {a.id for a in adr.active()}


def test_invalid_status_becomes_proposed():
    adr.record(adr.ADR("ADR-0400", "Weird", rationale="q", status="banana"))
    assert adr.get("ADR-0400").status == adr.PROPOSED


def test_filter_by_status():
    accepted = adr.all_adrs(status=adr.ACCEPTED)
    assert all(a.status == adr.ACCEPTED for a in accepted)
    assert len(accepted) >= 4  # the seeds


def test_record_is_fail_open():
    # A malformed ADR must not crash the registry.
    adr.record(adr.ADR(id="ADR-0500", title="t", rationale="r"))
    assert adr.get("ADR-0500")
