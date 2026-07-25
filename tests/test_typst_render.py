"""Stage-4 §3.3 — Typst PDF rendering: md→Typst conversion + fail-open.

The binary is baked into the pod image, so the actual compile can't run on the
dev box; we test the deterministic Markdown→Typst conversion and that
`render_pdf` fails open to None when the binary is absent (the dev-box state).
"""
from __future__ import annotations

from app.documents import typst_render as T


class TestConversion:
    def test_headings_map_to_equals(self):
        out = T.markdown_to_typst("# Title\n\n## Sub")
        assert "= Title" in out
        assert "== Sub" in out

    def test_bold_and_italic(self):
        out = T.markdown_to_typst("This is **bold** and *soft*.")
        assert "*bold*" in out       # Typst bold
        assert "_soft_" in out       # Typst italic

    def test_inline_code_preserved(self):
        out = T.markdown_to_typst("Run `kubectl get pods` now.")
        assert "`kubectl get pods`" in out

    def test_fenced_code_block_is_raw(self):
        md = "Example:\n\n```python\nprint('hi')\n```\n"
        out = T.markdown_to_typst(md)
        assert "```python" in out
        assert "print('hi')" in out  # verbatim, not escaped

    def test_bullet_and_numbered_lists(self):
        out = T.markdown_to_typst("- one\n- two\n\n1. first\n2. second")
        assert "- one" in out and "- two" in out
        assert "+ first" in out and "+ second" in out  # Typst enum

    def test_special_chars_escaped_in_prose(self):
        # A literal '#budget' / '$5' must not trigger Typst markup.
        out = T.markdown_to_typst("Total #budget is $5 for @ops.")
        assert "\\#budget" in out
        assert "\\$5" in out
        assert "\\@ops" in out

    def test_title_rendered_when_given(self):
        out = T.markdown_to_typst("body", title="My Report")
        assert "My Report" in out
        assert "#align(center)" in out

    def test_deterministic(self):
        md = "# H\n\ntext **b**\n\n- x"
        assert T.markdown_to_typst(md) == T.markdown_to_typst(md)


class TestFailOpen:
    def test_render_pdf_none_when_binary_absent(self, monkeypatch):
        monkeypatch.setattr(T, "_binary", lambda: None)
        assert T.render_pdf("# Hi", title="t") is None

    def test_render_pdf_none_on_blank_input(self, monkeypatch):
        monkeypatch.setattr(T, "_binary", lambda: "/usr/bin/typst")
        assert T.render_pdf("   ") is None

    def test_render_pdf_none_on_compile_failure(self, monkeypatch):
        monkeypatch.setattr(T, "_binary", lambda: "typst")

        class _Proc:
            returncode = 1
            stderr = b"syntax error"
        monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _Proc())
        assert T.render_pdf("# Hi") is None

    def test_available_reflects_binary(self, monkeypatch):
        monkeypatch.setattr(T, "_binary", lambda: None)
        assert T.available() is False
        monkeypatch.setattr(T, "_binary", lambda: "typst")
        assert T.available() is True

    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.documents, "typst", False, raising=False)
        assert T.enabled() is False
