"""Stage-6 §4.3 — band contracts precompute: structural table + pin cache."""
from __future__ import annotations

import pytest

from app.live import band_contract as BC


@pytest.fixture(autouse=True)
def _fresh():
    BC.reset_for_tests()
    yield
    BC.reset_for_tests()


class TestStructuralContract:
    def test_each_band_has_a_contract(self):
        for band in ("intern", "fresher", "junior", "mid", "senior", "lead",
                     "principal", "distinguished"):
            sc = BC.structural_contract(band)
            assert sc.band == band and sc.max_seconds > 0 and sc.sections

    def test_depth_scales_with_band(self):
        assert BC.structural_contract("intern").max_seconds \
            < BC.structural_contract("principal").max_seconds

    def test_intern_avoids_seniority_claims(self):
        avoid = " ".join(BC.structural_contract("intern").avoid).lower()
        assert "system design" in avoid or "team" in avoid

    def test_senior_must_include_a_tradeoff(self):
        mi = " ".join(BC.structural_contract("senior").must_include).lower()
        assert "tradeoff" in mi

    def test_unknown_band_falls_back_to_mid(self):
        assert BC.structural_contract("wizard").band == "mid"
        assert BC.structural_contract("").band == "mid"

    def test_as_dict_shape(self):
        d = BC.structural_contract("mid").as_dict()
        assert {"band", "depth", "ownership", "max_seconds", "sections",
                "must_include", "avoid"} <= set(d)


class TestPinCache:
    def test_pin_and_get(self):
        BC.pin("s1", real_band="senior", track="backend")
        p = BC.get_pinned("s1")
        assert p is not None and p.real_band == "senior"
        assert p.structural.band == "senior"        # structural = real band

    def test_pin_is_computed_once_and_reused(self):
        p1 = BC.pin("s1", real_band="mid", now=100.0)
        p2 = BC.get_pinned("s1")
        assert p2 is p1 and p2.pinned_at == 100.0   # same pinned object

    def test_get_miss_returns_none(self):
        assert BC.get_pinned("nope") is None

    def test_forget(self):
        BC.pin("s1", real_band="mid")
        BC.forget("s1")
        assert BC.get_pinned("s1") is None

    def test_unknown_real_band_uses_mid_structure(self):
        p = BC.pin("s1", real_band="wizard")
        assert p.structural.band == "mid"


class TestDirective:
    def test_directive_carries_band_shape(self):
        p = BC.pin("s1", real_band="senior")
        d = p.directive()
        assert "senior-level" in d
        assert "→" in d                             # the sections beats

    def test_frames_toward_target_band(self):
        p = BC.pin("s1", real_band="senior", target_band="lead")
        assert "lead role" in p.directive()
        assert "never overclaim" in p.directive().lower()

    def test_no_target_framing_when_bands_equal(self):
        p = BC.pin("s1", real_band="senior", target_band="senior")
        assert "frame toward" not in p.directive().lower()

    def test_includes_track_and_guidance(self):
        p = BC.pin("s1", real_band="mid", track="frontend",
                   guidance="Lean on demonstrated impact.")
        d = p.directive()
        assert "frontend interview" in d and "demonstrated impact" in d


class TestFlag:
    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "band_contracts", False, raising=False)
        assert BC.enabled() is False

    def test_enabled_reads_flag(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "band_contracts", True, raising=False)
        assert BC.enabled() is True
