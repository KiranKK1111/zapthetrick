"""Phase 1 — Structured Document Model (IR).

Pins the model parse, the structure-faithful Markdown round-trip, the new
model-driven HTML exporter, and that render_document routes 'html' through it."""
from __future__ import annotations

from app.documents.model import (
    CodeBlock, Diagram, DocumentModel, Heading, Image, ListBlock, Metadata,
    Paragraph, Quote, Section, Table,
    markdown_to_model, model_to_html, model_to_markdown,
)

SAMPLE = """# Design Doc

An intro paragraph about the system.

## Architecture

- gateway
- service
- database

Some prose here.

## Code

```python
def hi():
    return 1
```

## Data

| Name | Role |
|------|------|
| A | lead |
| B | dev |

## Flow

```mermaid
graph TD; A-->B
```

> a quoted note
"""


class TestParse:
    def test_sections_and_headings(self):
        m = markdown_to_model(SAMPLE)
        heads = [h for _, h in m.headings()]
        assert heads == ["Design Doc", "Architecture", "Code", "Data", "Flow"]
        assert m.metadata.title == "Design Doc"

    def test_block_types(self):
        m = markdown_to_model(SAMPLE)
        by_head = {s.heading: s for s in m.sections}
        assert isinstance(by_head["Architecture"].blocks[0], ListBlock)
        assert by_head["Architecture"].blocks[0].items == [
            "gateway", "service", "database"]
        assert isinstance(by_head["Code"].blocks[0], CodeBlock)
        assert by_head["Code"].blocks[0].language == "python"
        assert isinstance(by_head["Data"].blocks[0], Table)
        assert by_head["Data"].blocks[0].rows[0] == ["Name", "Role"]
        # A mermaid fence becomes a Diagram, not a CodeBlock.
        assert isinstance(by_head["Flow"].blocks[0], Diagram)
        assert isinstance(by_head["Flow"].blocks[1], Quote)

    def test_ordered_vs_unordered_list(self):
        m = markdown_to_model("1. first\n2. second\n\n- a\n- b\n")
        blocks = list(m.iter_blocks())
        assert isinstance(blocks[0], ListBlock) and blocks[0].ordered is True
        assert isinstance(blocks[1], ListBlock) and blocks[1].ordered is False

    def test_reading_time(self):
        long = "# T\n\n" + ("word " * 600) + "\n"
        assert markdown_to_model(long).metadata.reading_time_min == 3  # 600/200

    def test_lead_section_before_first_heading(self):
        m = markdown_to_model("intro line\n\n# H\n\nbody\n")
        assert m.sections[0].heading == "" and m.sections[0].level == 0
        assert isinstance(m.sections[0].blocks[0], Paragraph)


class TestRoundTrip:
    def test_structure_is_preserved(self):
        m1 = markdown_to_model(SAMPLE)
        m2 = markdown_to_model(model_to_markdown(m1))
        # Same section headings and same block-kind sequence per section.
        assert m1.headings() == m2.headings()
        for s1, s2 in zip(m1.sections, m2.sections):
            assert [b.kind for b in s1.blocks] == [b.kind for b in s2.blocks]

    def test_table_survives_round_trip(self):
        m2 = markdown_to_model(model_to_markdown(markdown_to_model(SAMPLE)))
        data = next(s for s in m2.sections if s.heading == "Data")
        assert data.blocks[0].rows[0] == ["Name", "Role"]
        assert ["A", "lead"] in data.blocks[0].rows


class TestHtml:
    def test_html_has_structure_and_escapes(self):
        m = markdown_to_model("# T\n\n<b>hi</b> & bye\n\n- x\n")
        html = model_to_html(m)
        assert html.startswith("<!doctype html>")
        assert "<title>T</title>" in html
        assert "&lt;b&gt;hi&lt;/b&gt; &amp; bye" in html   # escaped, not injected
        assert "<ul><li>x</li></ul>" in html

    def test_table_renders_as_html_table(self):
        html = model_to_html(markdown_to_model(SAMPLE))
        assert "<table>" in html and "<th>Name</th>" in html


class TestRenderDocumentHtml:
    def test_render_document_routes_html(self):
        from app.documents.generators import render_document, SUPPORTED_FORMATS
        assert "html" in SUPPORTED_FORMATS
        data, mime, ext = render_document(SAMPLE, "html", title="Design Doc")
        assert ext == "html" and mime.startswith("text/html")
        body = data.decode("utf-8")
        assert body.startswith("<!doctype html>") and "Architecture" in body

    def test_webpage_alias(self):
        from app.documents.generators import normalize_format
        assert normalize_format("webpage") == "html"
        assert normalize_format("htm") == "html"
