"""
BandSpecific taxonomy extensions for seniority calibration (app.live.calibration):
company numeric levels, the people-management ladder, the expanded per-family
tracks, the industry-vertical dimension, and capability-over-title framing.

Every new mapping/ladder/track/dimension is pinned here. The pre-existing
Violet-default and existing-band behaviour lives in test_live_calibration.py and
must stay green alongside these.
"""
from __future__ import annotations

from app.live import calibration as cal
from app.live import career as _career
from app.live.profile import build_profile


def _band(slug: str):
    return cal._BY_SLUG[slug]


# --------------------------------------------------------------------------- #
# 1. Company numeric internal levels → the right seniority band
# --------------------------------------------------------------------------- #
def test_google_levels():
    assert cal._band_from_title("Software Engineer, L3") == _band("junior").index
    assert cal._band_from_title("L5") == _band("senior").index
    assert cal._band_from_title("Google L6") == _band("lead").index
    assert cal._band_from_title("L7") == _band("principal").index


def test_amazon_ladder_is_distinct_from_google_when_named():
    # Bare L5 defaults to the Google-standard ladder (senior)...
    assert cal._band_from_title("L5") == _band("senior").index
    # ...but "Amazon L5" is SDE II → mid, and "Amazon L6" is Senior SDE → senior.
    assert cal._band_from_title("Amazon L5") == _band("mid").index
    assert cal._band_from_title("Amazon L6") == _band("senior").index
    assert cal._band_from_title("Amazon L7") == _band("principal").index


def test_meta_levels():
    assert cal._band_from_title("E3") == _band("junior").index
    assert cal._band_from_title("Meta E5") == _band("senior").index
    assert cal._band_from_title("E6") == _band("lead").index
    assert cal._band_from_title("E7") == _band("principal").index


def test_apple_ict_levels():
    assert cal._band_from_title("ICT2") == _band("junior").index
    assert cal._band_from_title("Apple ICT4") == _band("senior").index
    assert cal._band_from_title("ICT6") == _band("principal").index
    # ICT must NOT be misread as the generic IC ladder.
    assert cal._band_from_company_level("ICT4") == _band("senior").index


def test_microsoft_numeric_levels_require_company_context():
    assert cal._band_from_title("Microsoft 63") == _band("senior").index
    assert cal._band_from_title("Microsoft Level 65") == _band("principal").index
    # A bare 2-digit number with no Microsoft context is NOT treated as a level.
    assert cal._band_from_company_level("63") is None


def test_ibm_bands():
    assert cal._band_from_title("IBM Band 8") == _band("senior").index
    assert cal._band_from_title("Band 9") == _band("lead").index
    assert cal._band_from_title("Band 10") == _band("principal").index


def test_oracle_nvidia_ic_levels():
    assert cal._band_from_title("Oracle IC3") == _band("senior").index
    assert cal._band_from_title("IC4") == _band("lead").index
    assert cal._band_from_title("IC5") == _band("principal").index


def test_numeric_level_does_not_break_plain_titles():
    # No spurious level match inside ordinary words / tech tokens.
    assert cal._band_from_company_level("Senior Full Stack Developer, HTML5") is None
    assert cal._band_from_company_level("End-to-end (e2e) test engineer") is None


# --------------------------------------------------------------------------- #
# 2. People-management ladder + management-specific directive language
# --------------------------------------------------------------------------- #
def test_detect_management_tiers():
    assert cal.detect_management("Engineering Manager").slug == "em"
    assert cal.detect_management("Senior Engineering Manager").slug == "senior_em"
    assert cal.detect_management("Director of Engineering").slug == "director"
    assert cal.detect_management("VP of Engineering").slug == "vp"
    assert cal.detect_management("CTO").slug == "cto"
    # Most-specific wins: "Senior Director" is director-tier, not a bare match.
    assert cal.detect_management("Senior Director").slug == "director"


def test_ic_title_is_not_management():
    assert cal.detect_management("Senior Software Engineer") is None
    assert cal.detect_management("Staff Engineer") is None


def test_management_directive_language_is_people_leadership():
    profile = {"years_experience": "9", "current_role": "Engineering Manager"}
    c = cal.build_calibration(profile, {"job_role": "Engineering Manager"})
    assert c is not None and c.management is not None and c.management.slug == "em"
    d = cal.calibration_directive(c).lower()
    assert "management-track" in d
    assert "people leadership" in d
    assert "through others" in d


def test_ic_directive_has_no_management_block():
    profile = {"years_experience": "6", "current_role": "Senior Software Engineer"}
    c = cal.build_calibration(profile, {"job_role": "Senior Software Engineer"})
    assert c is not None and c.management is None
    assert "management-track" not in cal.calibration_directive(c).lower()


def test_management_skipped_on_override():
    # Manual override short-circuits inference → no management framing.
    profile = {"years_experience": "9", "current_role": "Engineering Manager"}
    c = cal.build_calibration(profile, {}, override="senior")
    assert c is not None and c.management is None


# --------------------------------------------------------------------------- #
# 3. Expanded per-family tracks
# --------------------------------------------------------------------------- #
def test_new_tracks_detected():
    cases = {
        "UX Designer": "design",
        "Product Designer": "design",
        "Technology Consultant": "consulting",
        "Solutions Engineer": "sales_eng",
        "Sales Engineer": "sales_eng",
        "Developer Advocate": "devrel",
        "Research Engineer": "research",
        "Applied Scientist": "research",
        "Network Engineer": "networking",
        "Embedded Software Engineer": "embedded",
        "Firmware Engineer": "embedded",
        "Game Engine Developer": "gaming",
        "Graphics Engineer": "gaming",
        "Blockchain Engineer": "blockchain",
        "Smart Contract Engineer": "blockchain",
        "SAP ABAP Developer": "enterprise",
        "Salesforce Developer": "enterprise",
    }
    for role, slug in cases.items():
        t = cal.detect_track(role)
        assert t is not None and t.slug == slug, f"{role} → {t and t.slug} (want {slug})"


def test_specific_track_still_beats_generic_engineer():
    # Regression: the specific-before-generic ordering must survive the additions.
    assert cal.detect_track("Senior AI Engineer", ["LLM", "RAG"]).slug == "ai_ml"
    assert cal.detect_track("", ["React", "TypeScript", "Flutter"]).slug == "frontend"
    assert cal.detect_track("Research Engineer").slug == "research"


def test_data_scientist_still_data_science():
    assert cal.detect_track("Senior Data Scientist").slug == "data_science"


# --------------------------------------------------------------------------- #
# 4. Industry-vertical dimension (domain × specialization × seniority × INDUSTRY)
# --------------------------------------------------------------------------- #
def test_detect_industry_from_jd():
    assert cal.detect_industry("Backend Engineer", "We build a payments platform for banking").slug == "fintech"
    assert cal.detect_industry("ML Engineer", "clinical data, HIPAA compliance").slug == "healthtech"
    assert cal.detect_industry("Engineer", "online learning platform / LMS").slug == "edtech"
    assert cal.detect_industry("Engineer", "high-scale ecommerce checkout").slug == "ecommerce"


def test_detect_industry_none_when_generic():
    assert cal.detect_industry("Software Engineer", "build features") is None
    assert cal.detect_industry("", None, []) is None


def test_industry_hint_in_directive():
    profile = {"years_experience": "4", "current_role": "Software Engineer"}
    org_ctx = {"job_role": "Backend Engineer",
               "job_description": "Own the payments and banking ledger; PCI compliance."}
    c = cal.build_calibration(profile, org_ctx)
    assert c is not None and c.industry is not None and c.industry.slug == "fintech"
    d = cal.calibration_directive(c).lower()
    assert "industry context" in d
    assert "fintech" in d
    # Must still refuse to fabricate domain years.
    assert "does not show" in d or "without claiming" in d


def test_industry_can_be_disabled_via_flag(monkeypatch):
    # `industry_context` is a getattr-with-enabling-default flag (not a field on
    # the frozen config model), so drive it by swapping the config object.
    import app.core.config_loader as _cl
    from types import SimpleNamespace
    monkeypatch.setattr(_cl, "cfg", SimpleNamespace(live=SimpleNamespace(industry_context=False)))
    org_ctx = {"job_role": "Backend Engineer", "job_description": "banking payments platform"}
    c = cal.build_calibration({"years_experience": "4"}, org_ctx)
    assert c is not None and c.industry is None


# --------------------------------------------------------------------------- #
# 5. Capability-over-title readiness framing (career.py consumed by calibration)
# --------------------------------------------------------------------------- #
def test_readiness_signal_public_helper():
    rich = build_profile({"skills": ["a", "b", "c", "d"] * 4,
                          "projects": [{"name": f"p{i}"} for i in range(8)],
                          "achievements": ["x", "y"]})
    assert _career.readiness_signal(rich) == "senior_ready"
    thin = build_profile({"skills": ["python"]})
    assert _career.readiness_signal(thin) == "entry"
    assert _career.readiness_signal(None) == "unknown"


def test_capability_directive_text():
    assert "capability over title" in _career.capability_directive("mid").lower()
    assert _career.capability_directive("unknown") == ""
    assert _career.capability_directive(None) == ""


def test_capability_framing_folded_into_directive():
    profile = {"years_experience": "2", "current_role": "Software Engineer",
               "skills": ["python", "aws", "llm", "rag", "kafka", "redis"],
               "projects": [{"name": f"p{i}"} for i in range(8)],
               "achievements": ["shipped X", "led Y"]}
    cp = build_profile(profile)
    c = cal.build_calibration(profile, {"job_role": "Senior Engineer"}, cp=cp)
    assert c is not None and c.readiness in {"entry", "mid", "senior_ready"}
    assert "capability over title" in cal.calibration_directive(c).lower()


def test_capability_framing_can_be_disabled_via_flag(monkeypatch):
    import app.core.config_loader as _cl
    from types import SimpleNamespace
    monkeypatch.setattr(_cl, "cfg", SimpleNamespace(live=SimpleNamespace(capability_framing=False)))
    profile = {"years_experience": "2", "current_role": "Software Engineer",
               "skills": ["python"], "projects": [{"name": "p"}]}
    c = cal.build_calibration(profile, {"job_role": "Senior Engineer"}, cp=build_profile(profile))
    assert c is not None and c.readiness is None


# --------------------------------------------------------------------------- #
# Fail-open guarantees for every new surface
# --------------------------------------------------------------------------- #
def test_new_surfaces_never_raise_on_garbage():
    assert cal._band_from_company_level(None) is None  # type: ignore[arg-type]
    assert cal.detect_management(None) is None          # type: ignore[arg-type]
    assert cal.detect_industry(None) is None            # type: ignore[arg-type]
    c = cal.build_calibration({"current_role": {"x": 1}}, {"job_description": 123})
    assert c is not None
    assert isinstance(cal.calibration_directive(c), str)
