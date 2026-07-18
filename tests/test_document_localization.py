"""Phase 7 — localization (furniture labels + generation directive)."""
from __future__ import annotations

from app.documents.localization import (
    is_rtl, is_supported, language_name, localization_directive,
    localize_labels, normalize_language, supported_languages,
)
from app.documents.model import markdown_to_model, model_to_markdown
from app.documents.structure import enrich


class TestNormalize:
    def test_code_name_and_endonym_all_resolve(self):
        assert normalize_language("fr") == "fr"
        assert normalize_language("French") == "fr"
        assert normalize_language("Français") == "fr"
        assert normalize_language("FRENCH") == "fr"

    def test_english_and_blank_are_none(self):
        # English needs no localization → the default path.
        assert normalize_language("en") is None
        assert normalize_language("English") is None
        assert normalize_language("") is None
        assert normalize_language(None) is None

    def test_unknown_is_none(self):
        assert normalize_language("klingon") is None
        assert is_supported("Spanish") is True
        assert is_supported("en") is False


class TestLabels:
    def test_spanish_labels_translated(self):
        labels = localize_labels("es")
        assert labels["toc"] == "Índice"
        assert labels["glossary"] == "Glosario"
        assert labels["figure"] == "Figura"

    def test_english_labels_default(self):
        labels = localize_labels("en")
        assert labels["toc"] == "Table of Contents"

    def test_unknown_falls_back_to_english(self):
        assert localize_labels("xx")["toc"] == "Table of Contents"

    def test_rtl_flag(self):
        assert is_rtl("Arabic") is True
        assert is_rtl("fr") is False


class TestDirective:
    def test_directive_names_language_and_conventions(self):
        d = localization_directive("de")
        assert "German" in d and "Deutsch" in d
        assert "decimal" in d.lower()

    def test_directive_empty_for_english(self):
        assert localization_directive("en") == ""
        assert localization_directive("") == ""

    def test_rtl_note_present_for_arabic(self):
        assert "right-to-left" in localization_directive("ar").lower()


class TestEnrichLocalizesFurniture:
    """The Phase-4 enrich pass localizes its injected sections via `lang`."""

    _MD = ("# Guide\n\nIntro.\n\n## Setup\n\nUses Kafka and Redis.\n\n"
           "## Deploy\n\nOn Kubernetes.\n\n## Verify\n\nCheck it.\n")

    def test_spanish_enrich_uses_spanish_toc(self):
        model = markdown_to_model(self._MD)
        out = model_to_markdown(enrich(model, lang="es"))
        assert "Índice" in out            # localized Table of Contents
        assert "Glosario" in out          # localized Glossary

    def test_default_enrich_stays_english(self):
        model = markdown_to_model(self._MD)
        out = model_to_markdown(enrich(model))
        assert "Table of Contents" in out
        assert "Índice" not in out

    def test_language_from_metadata_is_honored(self):
        model = markdown_to_model(self._MD)
        model.metadata.language = "fr"
        out = model_to_markdown(enrich(model))       # no explicit lang arg
        assert "Table des matières" in out


def test_supported_languages_lists_codes():
    langs = {l["code"] for l in supported_languages()}
    assert {"en", "es", "fr", "de", "ar"} <= langs
    assert language_name("fr") == "French"


class TestExportLocalization:
    """The `language` field on /export localizes the auto-generated furniture."""

    _MD = ("# Guía\n\nIntro.\n\n## Setup\n\nUses Kafka and Redis.\n\n"
           "## Deploy\n\nOn Kubernetes.\n\n## Verify\n\nCheck it.\n")

    def _client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from app.api.routes_documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_spanish_md_export_has_spanish_toc(self):
        c = self._client()
        r = c.post("/api/documents/export",
                   json={"content": self._MD, "format": "md", "language": "es"})
        assert r.status_code == 200
        body = r.content.decode("utf-8")
        assert "Índice" in body and "Glosario" in body

    def test_default_md_export_stays_english(self):
        c = self._client()
        r = c.post("/api/documents/export",
                   json={"content": self._MD, "format": "md"})
        assert r.status_code == 200
        body = r.content.decode("utf-8")
        assert "Table of Contents" in body and "Índice" not in body

    def test_languages_endpoint(self):
        c = self._client()
        r = c.get("/api/documents/languages")
        assert r.status_code == 200
        codes = {l["code"] for l in r.json()["languages"]}
        assert "es" in codes and "fr" in codes
