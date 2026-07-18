"""Phase 4 — document structure enhancers (TOC + glossary, model transforms)."""
from __future__ import annotations

from app.documents.model import markdown_to_model, model_to_html, model_to_markdown
from app.documents.structure import (
    build_glossary, build_list_of_exhibits, build_toc, detect_diagrams, enrich,
    number_exhibits, smart_appendix,
)

DOC = """# Payment Service

Intro about the service using Kafka and Redis.

## Overview

It exposes a REST API secured with JWT.

## Architecture

Kubernetes deploys the containers.

## Implementation

Details here.
"""


class TestToc:
    def test_toc_from_headings(self):
        toc = build_toc(markdown_to_model(DOC))
        assert toc is not None and toc.heading == "Table of Contents"
        items = toc.blocks[0].items
        assert any("Overview" in i for i in items)
        assert any("Architecture" in i for i in items)

    def test_no_toc_for_too_few_headings(self):
        assert build_toc(markdown_to_model("# Only\n\nbody\n")) is None


class TestGlossary:
    def test_defines_present_terms_only(self):
        g = build_glossary(markdown_to_model(DOC))
        assert g is not None and g.heading == "Glossary"
        joined = " ".join(g.blocks[0].items)
        assert "Kafka:" in joined and "JWT:" in joined and "Redis:" in joined
        assert "gRPC:" not in joined     # not mentioned → not defined

    def test_none_when_no_known_terms(self):
        assert build_glossary(markdown_to_model("# T\n\njust plain prose.\n")) is None


class TestEnrich:
    def test_inserts_toc_and_glossary(self):
        m = enrich(markdown_to_model(DOC))
        heads = [h for _, h in m.headings()]
        assert "Table of Contents" in heads and "Glossary" in heads
        # TOC sits after the lead/title, before the first real section.
        assert heads.index("Table of Contents") < heads.index("Overview")
        # Glossary is appended at the end.
        assert heads[-1] == "Glossary"

    def test_idempotent_when_already_present(self):
        m1 = enrich(markdown_to_model(DOC))
        m2 = enrich(m1)
        toc_count = sum(1 for _, h in m2.headings() if h == "Table of Contents")
        gloss_count = sum(1 for _, h in m2.headings() if h == "Glossary")
        assert toc_count == 1 and gloss_count == 1

    def test_original_model_unmutated(self):
        original = markdown_to_model(DOC)
        n_before = len(original.sections)
        enrich(original)
        assert len(original.sections) == n_before   # enrich returns a copy


EXHIBITS = """# Report

## Data

| Name | Role |
|------|------|
| A | lead |

## Second

| X | Y |
|---|---|
| 1 | 2 |

## Flow

Gateway -> Service -> Database
"""


class TestExhibitNumbering:
    def test_tables_and_figures_numbered(self):
        m = number_exhibits(detect_diagrams(markdown_to_model(EXHIBITS)))
        caps = [b.caption for b in m.iter_blocks() if getattr(b, "caption", "")]
        assert "Table 1" in caps and "Table 2" in caps
        assert "Figure 1" in caps            # the flow became a numbered figure

    def test_numbering_is_idempotent(self):
        m = number_exhibits(number_exhibits(markdown_to_model(EXHIBITS)))
        caps = [b.caption for b in m.iter_blocks() if getattr(b, "caption", "")]
        assert caps.count("Table 1") == 1    # not "Table 1: Table 1"
        assert all(c.count("Table") <= 1 for c in caps)

    def test_list_of_exhibits(self):
        m = number_exhibits(detect_diagrams(markdown_to_model(EXHIBITS)))
        lx = build_list_of_exhibits(m)
        assert lx is not None and lx.heading == "List of Figures & Tables"
        assert any("Table 1" in i for i in lx.blocks[0].items)

    def test_no_exhibit_list_when_too_few(self):
        m = number_exhibits(markdown_to_model("# T\n\n| A |\n|---|\n| 1 |\n"))
        assert build_list_of_exhibits(m) is None


class TestAutoDiagram:
    def test_arrow_flow_becomes_diagram(self):
        m = detect_diagrams(markdown_to_model("# T\n\nUser -> Gateway -> DB\n"))
        diagrams = [b for b in m.iter_blocks() if b.kind == "diagram"]
        assert len(diagrams) == 1
        assert diagrams[0].diagram_kind == "mermaid"
        assert "flowchart" in diagrams[0].source and "Gateway" in diagrams[0].source

    def test_unicode_arrows(self):
        m = detect_diagrams(markdown_to_model("# T\n\nA → B → C\n"))
        assert any(b.kind == "diagram" for b in m.iter_blocks())

    def test_prose_is_not_mangled(self):
        # A normal sentence with one arrow / two parts must NOT become a diagram.
        prose = "# T\n\nThe request goes to the server and back.\n"
        m = detect_diagrams(markdown_to_model(prose))
        assert not any(b.kind == "diagram" for b in m.iter_blocks())


class TestSmartAppendix:
    def test_config_block_moved_to_appendix(self):
        md = "# T\n\nintro\n\n```yaml\ndb: pg\nhost: local\n```\n"
        m = smart_appendix(markdown_to_model(md))
        assert any(s.heading == "Appendix" for s in m.sections)
        # A reference is left in the body.
        body = model_to_markdown(m)
        assert "See Appendix" in body and "db: pg" in body   # moved, not lost

    def test_short_code_stays_inline(self):
        md = "# T\n\n```python\nx = 1\n```\n"
        m = smart_appendix(markdown_to_model(md))
        assert not any(s.heading == "Appendix" for s in m.sections)

    def test_idempotent_when_appendix_exists(self):
        md = "# T\n\n```yaml\na: 1\n```\n\n## Appendix\n\nexisting\n"
        m = smart_appendix(markdown_to_model(md))
        assert sum(1 for s in m.sections if s.heading == "Appendix") == 1


class TestCaptionsAndAnchors:
    def test_caption_renders_in_markdown_and_html(self):
        m = number_exhibits(markdown_to_model(EXHIBITS))
        assert "*Table 1*" in model_to_markdown(m)
        html = model_to_html(m)
        assert "<figcaption>Table 1</figcaption>" in html

    def test_html_heading_anchors_and_page_breaks(self):
        html = model_to_html(markdown_to_model(EXHIBITS))
        assert 'id="data"' in html and 'id="second"' in html
        assert "page-break-inside" in html

    def test_html_toc_is_clickable(self):
        html = model_to_html(enrich(markdown_to_model(EXHIBITS)))
        assert 'class="toc"' in html and 'href="#data"' in html


class TestAssets:
    def test_asset_registry(self):
        m = detect_diagrams(markdown_to_model("# T\n\nA -> B -> C\n"))
        assets = m.assets()
        assert len(assets) == 1 and assets[0].kind == "diagram"


class TestRenderIntegration:
    def test_html_gate_on_by_default(self):
        from app.documents.generators import render_document
        # Phase 4 is now ACTIVE by default → a multi-heading doc gains a TOC.
        html = render_document(DOC, "html", title="Payment Service")[0].decode()
        assert "Table of Contents" in html

    def test_html_gate_respects_flag_off(self, monkeypatch):
        from app.documents.generators import render_document
        monkeypatch.setattr("app.documents.structure.auto_structure_enabled",
                            lambda: False)
        html = render_document(DOC, "html", title="Payment Service")[0].decode()
        assert "Table of Contents" not in html

    def test_enriched_html_has_toc(self):
        html = model_to_html(enrich(markdown_to_model(DOC)))
        assert "Table of Contents" in html and "Glossary" in html

    def test_full_enrich_pipeline(self):
        m = enrich(markdown_to_model(EXHIBITS))
        ordered = [h for _, h in m.headings()]
        assert {"Table of Contents", "Data",
                "List of Figures & Tables"} <= set(ordered)
        # TOC precedes the first real section.
        assert ordered.index("Table of Contents") < ordered.index("Data")


class TestRendererMigration:
    """The prose renderers (PDF/DOCX/PPTX/TXT/MD) now go through the model, so
    they get the same structure enrichment + branding as HTML."""

    def test_branding_reaches_prose_formats(self):
        from app.documents.generators import render_document
        from app.documents.model import ExportSettings
        es = ExportSettings(header="ACME", footer="Confidential ©",
                            confidentiality="Internal", author="Jane")
        txt = render_document(DOC, "txt", title="T",
                              export_settings=es)[0].decode()
        assert "ACME" in txt and "Internal" in txt and "Jane" in txt
        assert "Confidential" in txt
        # Binary formats still produce valid non-trivial output.
        for fmt in ("docx", "pdf"):
            data = render_document(DOC, fmt, export_settings=es)[0]
            assert len(data) > 200

    def test_no_branding_no_change(self, monkeypatch):
        from app.documents.generators import render_document
        # With enrichment held OFF and no settings → legacy (un-branded) output.
        monkeypatch.setattr("app.documents.structure.auto_structure_enabled",
                            lambda: False)
        txt = render_document(DOC, "txt")[0].decode()
        assert "ACME" not in txt and "Table of Contents" not in txt

    def test_enrichment_reaches_prose_when_on(self, monkeypatch):
        # auto_structure ON → the prose markdown gains a TOC + glossary before
        # the (unchanged) renderer sees it.
        import app.documents.generators as G
        monkeypatch.setattr("app.documents.structure.auto_structure_enabled",
                            lambda: True)
        enriched = G._enrich_prose_markdown(DOC, "Payment Service")
        assert "Table of Contents" in enriched and "Glossary" in enriched

    def test_enrichment_off_is_byte_identical(self, monkeypatch):
        import app.documents.generators as G
        monkeypatch.setattr("app.documents.structure.auto_structure_enabled",
                            lambda: False)
        assert G._enrich_prose_markdown(DOC, "T") == DOC
