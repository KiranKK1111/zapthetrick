"""Phase 7 — export profiles / persona engine, branding, and prefs helper."""
from __future__ import annotations

import pytest

from app.documents.model import ExportSettings, markdown_to_model, model_to_html
from app.documents.preferences import export_settings_from_prefs
from app.documents.profiles import (
    Audience, detect_audience, persona_directive, persona_for,
)



@pytest.fixture(autouse=True)
def _deterministic_classifiers(monkeypatch):
    """Pin the DETERMINISTIC fallback. detect_audience / detect_document_goal /
    detect_project_type are SEMANTIC-first (gates.classify), which is warm in the
    full suite and returns valid-but-different classes than the regex these tests
    pin. The semantic mechanism is covered in test_semantic_gates; here we test
    the fallback, so disable the embedding classifier."""
    import app.semantics.gates as _g
    monkeypatch.setattr(_g, "classify", lambda *a, **k: None)

class TestAudienceDetection:
    @pytest.mark.parametrize("text,aud", [
        ("write this for my manager", Audience.MANAGER),
        ("a brief for the CTO", Audience.EXECUTIVE),
        ("send to leadership", Audience.EXECUTIVE),
        ("docs for the engineering team", Audience.DEVELOPER),
        ("a summary for the client", Audience.CLIENT),
        ("explain for beginners", Audience.STUDENT),
        ("for HR", Audience.HR),
        ("just explain how kafka works", Audience.GENERAL),
    ])
    def test_detect(self, text, aud):
        assert detect_audience(text) == aud


class TestPersona:
    def test_directive_for_audience(self):
        d = persona_directive("for the CTO")
        assert "executive" in d and "high-level" in d

    def test_general_has_no_directive(self):
        assert persona_directive("explain recursion") == ""

    def test_persona_fields(self):
        p = persona_for(Audience.DEVELOPER)
        assert p.detail == "deep" and "technical" in p.tone
        assert set(p.as_dict()) == {"audience", "tone", "detail", "emphasis"}


class TestBranding:
    def test_html_honors_export_settings(self):
        m = markdown_to_model("# Report\n\nbody")
        m.export = ExportSettings(
            header="ACME Corp", footer="© 2026 ACME",
            primary_color="#4f46e5", confidentiality="Confidential",
            author="Jane Doe", logo_url="https://x/logo.png")
        html = model_to_html(m)
        assert "ACME Corp" in html and "© 2026 ACME" in html
        assert "#4f46e5" in html and "Confidential" in html
        assert "Jane Doe" in html and 'src="https://x/logo.png"' in html

    def test_no_branding_when_unset(self):
        html = model_to_html(markdown_to_model("# T\n\nbody"))
        assert 'class="brand"' not in html and "Confidential" not in html

    def test_render_document_html_applies_branding(self):
        from app.documents.generators import render_document
        es = ExportSettings(header="MyCo", primary_color="#111")
        data, mime, ext = render_document("# T\n\nbody", "html",
                                          export_settings=es)
        body = data.decode()
        assert ext == "html" and "MyCo" in body and "#111" in body


class TestPrefsHelper:
    def test_export_settings_from_prefs(self):
        es = export_settings_from_prefs(
            {"branding": {"header": "H", "primary_color": "#abc",
                          "author": "A"}})
        assert es.header == "H" and es.primary_color == "#abc" and es.author == "A"

    def test_empty_prefs_gives_blank_settings(self):
        es = export_settings_from_prefs({})
        assert es.header == "" and es.primary_color == ""
        # Robust to junk.
        assert export_settings_from_prefs(None).header == ""
        assert export_settings_from_prefs({"branding": "nope"}).header == ""
