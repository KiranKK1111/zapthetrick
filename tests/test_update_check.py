"""Tests for the version-manifest + update-check foundation (Phase 1 #24).

Pure decision logic — no network. Also asserts the wiring: main.py's `/` endpoint
reports APP_VERSION (single source of truth), kept in sync with the FE pubspec.
"""
from __future__ import annotations

import pathlib

import pytest

from app.core.update_check import (
    APP_VERSION,
    ReleaseManifest,
    UpdateStatus,
    check_for_update,
    compare_versions,
    parse_version,
)


@pytest.mark.parametrize("v,expected", [
    ("1.2.3", (1, 2, 3)),
    ("v0.2.0", (0, 2, 0)),
    ("2.0", (2, 0, 0)),
    ("1.2.3-beta+build9", (1, 2, 3)),
    ("", None),
    ("1.x.0", None),
    ("not.a.version", None),
])
def test_parse_version(v, expected):
    assert parse_version(v) == expected


@pytest.mark.parametrize("a,b,expected", [
    ("1.0.0", "1.0.1", -1),
    ("1.2.0", "1.2.0", 0),
    ("2.0.0", "1.9.9", 1),
    ("0.2.0", "0.10.0", -1),   # numeric, not lexical
    ("bad", "1.0.0", None),
])
def test_compare_versions(a, b, expected):
    assert compare_versions(a, b) == expected


def test_up_to_date():
    m = ReleaseManifest(latest="0.2.0", minimum_supported="0.1.0")
    r = check_for_update("0.2.0", m)
    assert r.status is UpdateStatus.UP_TO_DATE
    assert not r.update_available


def test_update_available():
    m = ReleaseManifest(latest="0.3.0", minimum_supported="0.1.0",
                        notes="new viewers", url="https://example/dl")
    r = check_for_update("0.2.0", m)
    assert r.status is UpdateStatus.UPDATE_AVAILABLE
    assert r.update_available
    assert r.notes == "new viewers" and r.url


def test_update_required_below_minimum():
    m = ReleaseManifest(latest="0.3.0", minimum_supported="0.2.0")
    r = check_for_update("0.1.5", m)
    assert r.status is UpdateStatus.UPDATE_REQUIRED
    assert r.update_available  # required counts as available


def test_unknown_on_bad_version():
    m = ReleaseManifest(latest="0.3.0")
    assert check_for_update("garbage", m).status is UpdateStatus.UNKNOWN


def test_manifest_from_dict():
    m = ReleaseManifest.from_dict({"latest": "1.0.0", "channel": "beta"})
    assert m.latest == "1.0.0" and m.channel == "beta"
    assert m.minimum_supported == "0.0.0"  # default


def test_wiring_main_uses_app_version():
    # main.py must report APP_VERSION, not a hardcoded literal.
    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
    assert '"version": APP_VERSION' in src, "main.py '/' must report APP_VERSION."
    assert '"version": "0.2.0"' not in src, "main.py must not hardcode the version."


def test_check_for_update_is_now_called_in_prod():
    """P1 #24: the roadmap flagged `check_for_update` as never called. It is now
    invoked by both the maintenance loop and the /api/health/update endpoint."""
    from app.obs import maintenance
    rep = maintenance.run_maintenance_once(run_benchmark=False)
    assert rep["update"].get("status") == "up_to_date"

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "app" / "api" / "routes_jobs.py").read_text(encoding="utf-8")
    assert "check_for_update" in src, "the update endpoint must call check_for_update"


def test_update_endpoint_applies_rollout():
    """The endpoint layers staged-rollout on top of the version decision."""
    from app.core.rollout import rollout_decision
    # A newer version, device in a 100% rollout → offered.
    d = rollout_decision("dev-1", "9.9.9", APP_VERSION, percent=100)
    assert d["offer"] is True
    # A blocked (rolled-back) version → never offered.
    d2 = rollout_decision("dev-1", "9.9.9", APP_VERSION, percent=100,
                          blocked={"9.9.9"})
    assert d2["offer"] is False


def test_app_version_matches_fe_pubspec():
    # Single source of truth: BE APP_VERSION should match the FE pubspec version.
    pub = pathlib.Path(__file__).resolve().parents[2] / "zapthetrick_fe" / "pubspec.yaml"
    if not pub.exists():
        pytest.skip("FE pubspec not present in this checkout")
    line = next((l for l in pub.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("version:")), "")
    fe_version = line.split(":", 1)[1].strip().split("+")[0] if ":" in line else ""
    assert fe_version == APP_VERSION, (
        f"BE APP_VERSION={APP_VERSION} != FE pubspec version={fe_version} — keep them in sync."
    )
