"""Stage-4 §3.3 — bounded visual-QA repair loop.

The repair orchestrator rewrites the source to address HIGH-severity findings,
re-renders via an injected render_fn, re-checks, and keeps the fewest-issues
bytes. Rasterization + the real VLM are stubbed; only the loop logic is tested.
"""
from __future__ import annotations

import asyncio

import pytest

from app.verify import visual_qa as V
from app.verify.visual_qa import VisualIssue, VisualQAReport, repair_visual_qa


def _run(coro):
    return asyncio.run(coro)


def _rep(*issues, ok=None):
    hi = [i for i in issues if i.severity == "high"]
    return VisualQAReport(ok=(ok if ok is not None else not hi),
                          issues=list(issues), ran=True, page_count=1)


@pytest.fixture(autouse=True)
def _stub_rewrite(monkeypatch):
    # The LLM rewrite just tags the content so we can assert a re-render happened.
    async def fake_rewrite(content, findings, rubric):
        return content + "\n\n## Skills\nPython"
    monkeypatch.setattr(V, "_rewrite_source", fake_rewrite)


class TestRepairLoop:
    def test_fixes_high_issue_and_keeps_better(self):
        bad = _rep(VisualIssue("missing", "no skills section", "high"))
        # Second render is clean → accepted.
        reports = iter([_rep(ok=True)])

        async def fake_vqa(pdf, rub, *, describe_fn, page_hint=None):
            return next(reports)
        # Patch the module-level visual_qa the loop calls.
        import app.verify.visual_qa as mod
        orig = mod.visual_qa
        mod.visual_qa = fake_vqa
        try:
            rendered = []

            def render_fn(c):
                rendered.append(c)
                return b"%PDF-new"
            out_bytes, out_rep = _run(repair_visual_qa(
                "# CV", b"%PDF-old", bad, "include a skills section",
                render_fn=render_fn, describe_fn=None, page_hint=1,
                max_rounds=1))
        finally:
            mod.visual_qa = orig
        assert out_bytes == b"%PDF-new"       # accepted the improved render
        assert out_rep.ok is True
        assert rendered and "## Skills" in rendered[0]

    def test_no_improvement_keeps_original(self):
        bad = _rep(VisualIssue("missing", "no skills", "high"))

        async def fake_vqa(pdf, rub, *, describe_fn, page_hint=None):
            return _rep(VisualIssue("missing", "still no skills", "high"))
        import app.verify.visual_qa as mod
        orig = mod.visual_qa
        mod.visual_qa = fake_vqa
        try:
            out_bytes, out_rep = _run(repair_visual_qa(
                "# CV", b"%PDF-old", bad, "skills",
                render_fn=lambda c: b"%PDF-new", describe_fn=None,
                max_rounds=2))
        finally:
            mod.visual_qa = orig
        assert out_bytes == b"%PDF-old"       # regression rejected
        assert out_rep is bad

    def test_zero_rounds_is_noop(self):
        bad = _rep(VisualIssue("missing", "x", "high"))
        out_bytes, out_rep = _run(repair_visual_qa(
            "# CV", b"%PDF-old", bad, "r",
            render_fn=lambda c: b"NOPE", describe_fn=None, max_rounds=0))
        assert out_bytes == b"%PDF-old" and out_rep is bad

    def test_render_failure_is_fail_open(self):
        bad = _rep(VisualIssue("overflow", "x", "high"))

        def boom(c):
            raise RuntimeError("render exploded")
        out_bytes, out_rep = _run(repair_visual_qa(
            "# CV", b"%PDF-old", bad, "r",
            render_fn=boom, describe_fn=None, max_rounds=1))
        assert out_bytes == b"%PDF-old" and out_rep is bad

    def test_async_render_fn_supported(self):
        bad = _rep(VisualIssue("missing", "x", "high"))

        async def fake_vqa(pdf, rub, *, describe_fn, page_hint=None):
            return _rep(ok=True)
        import app.verify.visual_qa as mod
        orig = mod.visual_qa
        mod.visual_qa = fake_vqa
        try:
            async def arender(c):
                return b"%PDF-async"
            out_bytes, out_rep = _run(repair_visual_qa(
                "# CV", b"%PDF-old", bad, "r",
                render_fn=arender, describe_fn=None, max_rounds=1))
        finally:
            mod.visual_qa = orig
        assert out_bytes == b"%PDF-async" and out_rep.ok is True


class TestFindingsText:
    def test_high_first_and_marked(self):
        r = _rep(VisualIssue("style", "nit", "low"),
                 VisualIssue("missing", "big", "high"))
        txt = V._findings_text(r)
        assert txt.index("MUST FIX") < txt.index("consider")
        assert "big" in txt and "nit" in txt
