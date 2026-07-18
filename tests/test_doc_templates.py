"""Document design templates (P4 #21) — ATS-friendly resume rendering."""
from __future__ import annotations

from app.documents import templates as T

SECTIONS = {
    "name": "Ada Lovelace",
    "contact": ["ada@x.io", "London"],
    "summary": "Analytical engine pioneer.",
    "experience": [
        {"role": "Mathematician", "company": "Analytical Engine",
         "dates": "1842–1843", "bullets": ["Wrote the first algorithm"]},
    ],
    "skills": ["Math", "Algorithms", "Writing"],
    "education": [{"degree": "Self-taught", "school": "—"}],
}


def test_lists_three_ats_safe_templates():
    ids = {t["id"] for t in T.list_templates()}
    assert {"classic", "modern", "compact"} <= ids
    assert all(t["ats_safe"] for t in T.list_templates())


def test_render_has_name_sections_and_content():
    md = T.render_resume(SECTIONS, "classic")
    assert md.startswith("# Ada Lovelace")
    assert "## Experience" in md
    assert "Wrote the first algorithm" in md
    assert "ada@x.io" in md


def test_modern_uses_uppercase_headings():
    md = T.render_resume(SECTIONS, "modern")
    assert "## SUMMARY" in md and "## EXPERIENCE" in md


def test_order_differs_by_template():
    classic = T.render_resume(SECTIONS, "classic")
    modern = T.render_resume(SECTIONS, "modern")
    # modern puts skills before experience; classic puts experience first
    assert modern.index("SKILLS") < modern.index("EXPERIENCE")
    assert classic.index("Experience") < classic.index("Education")


def test_unknown_template_falls_back_to_classic():
    assert T.render_resume(SECTIONS, "nope").startswith("# Ada Lovelace")


def test_extra_sections_are_preserved():
    md = T.render_resume({**SECTIONS, "Certifications": ["AWS SA"]}, "classic")
    assert "## Certifications" in md and "AWS SA" in md


# ── BUG 4: templates are reachable — markdown round-trip + export/preview ────
RESUME_MD = """# Ada Lovelace

ada@x.io · London

## Professional Summary

Analytical engine pioneer.

## Work Experience

**Mathematician · Analytical Engine** — 1842–1843

- Wrote the first algorithm
- Published the notes

## Technical Skills

- Math
- Algorithms

## Education

**Self-taught**
"""


class TestSectionsFromMarkdown:
    def test_parses_a_real_resume(self):
        s = T.sections_from_markdown(RESUME_MD)
        assert s["name"] == "Ada Lovelace"
        assert "ada@x.io" in s["contact"] and "London" in s["contact"]
        assert s["summary"].startswith("Analytical engine")
        assert s["skills"] == ["Math", "Algorithms"]
        assert s["experience"][0]["bullets"] == [
            "Wrote the first algorithm", "Published the notes"]

    def test_non_resume_content_is_not_parsed(self):
        assert T.sections_from_markdown("# Design\n\n## Database\n\nMySQL.") == {}

    def test_unsupported_blocks_bail_out(self):
        # A table can't be laid out ATS-safely → refuse rather than drop content.
        md = RESUME_MD + "\n## Skills\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
        assert T.sections_from_markdown(md) == {}


class TestApplyTemplate:
    def test_relays_out_the_resume(self):
        out = T.apply_template(RESUME_MD, "modern")
        assert "## SKILLS" in out and "## EXPERIENCE" in out
        assert out.index("SKILLS") < out.index("EXPERIENCE")   # modern order
        assert "Wrote the first algorithm" in out              # content kept
        assert "ada@x.io" in out

    def test_no_template_is_a_no_op(self):
        assert T.apply_template(RESUME_MD, None) == RESUME_MD
        assert T.apply_template(RESUME_MD, "") == RESUME_MD

    def test_unknown_template_is_a_no_op(self):
        assert T.apply_template(RESUME_MD, "fancy") == RESUME_MD

    def test_non_resume_content_untouched(self):
        doc = "# Design\n\n## Database\n\nMySQL."
        assert T.apply_template(doc, "classic") == doc


class TestGeneratorIntegration:
    def test_render_document_without_template_is_unchanged(self):
        # The default path must be byte-identical to the legacy one.
        from app.documents.generators import render_document
        a, _, _ = render_document(RESUME_MD, "md")
        b, _, _ = render_document(RESUME_MD, "md", template=None)
        assert a == b
        assert b"Professional Summary" in a and b"## SKILLS" not in a

    def test_render_document_with_template(self):
        from app.documents.generators import render_document
        data, _, ext = render_document(RESUME_MD, "md", template="modern")
        assert ext == "md" and b"## SKILLS" in data

    def test_docx_still_renders_through_a_template(self):
        from app.documents.generators import render_document
        data, _, ext = render_document(RESUME_MD, "docx", title="CV",
                                       template="compact")
        assert ext == "docx" and data[:2] == b"PK"


class TestTemplateRoutes:
    def _client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from app.api.routes_documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_list_templates_endpoint(self):
        js = self._client().get("/api/documents/templates").json()
        ids = {t["id"] for t in js["templates"]}
        assert {"classic", "modern", "compact"} <= ids
        assert all(t["ats_safe"] for t in js["templates"])

    def test_export_with_template(self):
        r = self._client().post(
            "/api/documents/export",
            json={"content": RESUME_MD, "format": "md", "template": "modern"})
        assert r.status_code == 200
        assert b"## SKILLS" in r.content

    def test_export_without_template_is_the_default_path(self):
        r = self._client().post(
            "/api/documents/export",
            json={"content": RESUME_MD, "format": "md"})
        assert r.status_code == 200
        assert b"## SKILLS" not in r.content        # no re-layout
        assert b"## Professional Summary" in r.content

    def test_export_rejects_an_unknown_template(self):
        r = self._client().post(
            "/api/documents/export",
            json={"content": RESUME_MD, "format": "md", "template": "fancy"})
        assert r.status_code == 400
