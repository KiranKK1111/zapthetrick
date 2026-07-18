"""Seniority-band calibration for live interview answers (app.live.calibration)."""
from __future__ import annotations

from app.live import calibration as cal
from app.live.profile import build_profile


def _band(slug: str):
    return cal._BY_SLUG[slug]


# --------------------------------------------------------------------------- #
# Real-band classification
# --------------------------------------------------------------------------- #
def test_fresher_from_zero_years():
    b, _ = cal.classify_real_band({"years_experience": 0, "current_role": ""})
    assert b.index <= 2


def test_senior_from_years_and_title():
    b, _ = cal.classify_real_band(
        {"years_experience": "7", "current_role": "Senior Software Engineer"})
    assert b.index >= 4


def test_title_lifts_but_is_bounded_over_years():
    # 1 year of experience but a "Principal" title → don't jump straight to
    # principal on a title alone; capped to one band over the years estimate.
    b, _ = cal.classify_real_band(
        {"years_experience": "1", "current_role": "Principal Engineer"})
    assert b.index <= _band("mid").index


def test_manual_override_wins():
    b, sig = cal.classify_real_band(
        {"years_experience": "15", "current_role": "Principal Engineer"},
        override="fresher")
    assert b.slug == "fresher"
    assert sig.get("override") == "fresher"


def test_override_auto_is_ignored():
    b, sig = cal.classify_real_band(
        {"years_experience": "6", "current_role": "Senior Engineer"},
        override="auto")
    assert "override" not in sig
    assert b.index >= 4


def test_no_signal_defaults_to_fresher_floor():
    b, _ = cal.classify_real_band({})
    assert b.slug == "fresher"


def test_years_string_with_plus_parses():
    assert cal._parse_years("5+") == 5.0
    assert cal._parse_years("3-4 years") == 3.0
    assert cal._parse_years(None) is None


# --------------------------------------------------------------------------- #
# Track detection
# --------------------------------------------------------------------------- #
def test_track_ai_beats_generic_engineer():
    t = cal.detect_track("Senior AI Engineer", ["Python", "LLM", "RAG"])
    assert t is not None and t.slug == "ai_ml"


def test_track_from_skills_when_role_blank():
    t = cal.detect_track("", ["React", "TypeScript", "Flutter"])
    assert t is not None and t.slug == "frontend"


def test_track_none_when_unknown():
    assert cal.detect_track("", []) is None


# --------------------------------------------------------------------------- #
# Target band + full calibration + directive
# --------------------------------------------------------------------------- #
def test_target_band_from_job_role():
    t = cal.classify_target_band({"job_role": "Staff Engineer"})
    assert t is not None and t.index >= _band("lead").index


def test_build_calibration_and_directive_gap_up():
    profile = {"years_experience": "2", "current_role": "Software Engineer",
               "skills": ["Python", "AWS", "LLM", "RAG"]}
    org_ctx = {"job_role": "Senior AI Engineer"}
    c = cal.build_calibration(profile, org_ctx, cp=build_profile(profile))
    assert c is not None
    d = cal.calibration_directive(c)
    assert "SENIORITY CALIBRATION" in d
    # Real band is below the target → the "frame toward higher expectations,
    # without claiming" bridge must appear, and truthfulness must be stated.
    assert "without claiming" in d.lower() or "growth trajectory" in d.lower()
    assert "truthful" in d.lower()
    assert c.track is not None and c.track.slug == "ai_ml"


def test_directive_overqualified_case():
    profile = {"years_experience": "10", "current_role": "Principal Engineer"}
    org_ctx = {"job_role": "Software Engineer II"}
    c = cal.build_calibration(profile, org_ctx)
    d = cal.calibration_directive(c)
    assert "exceeds the target" in d.lower()


def test_directive_empty_on_none():
    assert cal.calibration_directive(None) == ""


def test_build_calibration_never_raises_on_garbage():
    c = cal.build_calibration({"years_experience": {"weird": 1}},
                              {"job_role": 123})
    assert c is not None
    assert isinstance(cal.calibration_directive(c), str)


def test_directive_always_ends_professional():
    c = cal.build_calibration({"years_experience": 0}, {})
    d = cal.calibration_directive(c)
    assert "professionalism" in d.lower()


# --------------------------------------------------------------------------- #
# Tuning: expanded title synonyms
# --------------------------------------------------------------------------- #
def test_mts_ladder_titles():
    # "Member of Technical Staff" ladder maps onto the seniority tiers.
    assert cal._band_from_title("Senior Member of Technical Staff") is not None
    assert cal._band_from_title("SMTS") == _band("senior").index
    assert cal._band_from_title("PMTS") == _band("principal").index


def test_founding_and_lead_synonyms():
    assert cal._band_from_title("Founding Engineer") == _band("senior").index
    assert cal._band_from_title("Engineering Lead") == _band("lead").index
    assert cal._band_from_title("Senior Engineering Manager") == _band("principal").index


def test_sde_variants():
    assert cal._band_from_title("SDE III") == _band("senior").index
    assert cal._band_from_title("SDE2") == _band("mid").index
