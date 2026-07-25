"""Stage-4 §3.3 Component G — requirement rubric + visual-QA loop.

The rubric extracts a document request's explicit asks; the visual-QA loop
rasterizes a rendered PDF and has the (injected) VLM critique it against that
rubric. The VLM call is dependency-injected so `app.verify` stays vision-free;
here it's stubbed, and a real one-page PDF is built with PyMuPDF.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.documents.rubric import Rubric, extract_rubric
from app.verify import visual_qa as vq


def _run(coro):
    return asyncio.run(coro)


def _pdf(text: str = "Hello Visual QA", pages: int = 1) -> bytes:
    import fitz
    d = fitz.open()
    for _ in range(pages):
        p = d.new_page()
        p.insert_text((72, 72), text)
    out = d.tobytes()
    d.close()
    return out


# --------------------------------------------------------------------------- #
class TestRubric:
    def test_pages_word_and_digit(self):
        assert extract_rubric("a one page resume").pages == 1
        assert extract_rubric("write a 2 page report").pages == 2
        assert extract_rubric("two-page cover letter").pages == 2

    def test_type_style_ats(self):
        r = extract_rubric("a modern ATS-friendly resume")
        assert r.type == "resume" and r.style == "modern" and r.ats is True

    def test_minimalist_normalizes(self):
        assert extract_rubric("a minimalist cv").style == "minimal"

    def test_multiple_must_include_sections(self):
        r = extract_rubric(
            "make a resume, include a skills section and a projects section")
        assert "skills" in r.must_include and "projects" in r.must_include

    def test_section_not_captured_without_include_verb(self):
        # A stray "section" on a plain answer shouldn't become a requirement.
        assert extract_rubric("explain the skills section of a resume") \
            .must_include == []

    def test_empty_request(self):
        r = extract_rubric("")
        assert r.is_empty() and r.as_dict()["pages"] is None

    def test_checklist_text(self):
        txt = extract_rubric("a 1 page ATS resume").checklist_text()
        assert "exactly 1" in txt and "ATS" in txt


class TestRasterize:
    def test_real_pdf_rasterizes(self):
        pngs = vq.rasterize_pdf(_pdf())
        assert len(pngs) == 1
        assert pngs[0].startswith("iVBOR")   # PNG base64 signature

    def test_multipage(self):
        assert len(vq.rasterize_pdf(_pdf(pages=3))) == 3

    def test_bad_bytes_fail_open(self):
        assert vq.rasterize_pdf(b"not a pdf") == []


class TestVisualQA:
    def _describe(self, reply):
        async def fn(images, prompt):
            assert images and isinstance(prompt, str)
            return reply
        return fn

    def test_pass_report(self):
        reply = json.dumps({"meets_rubric": True, "page_count": 1, "issues": []})
        rep = _run(vq.visual_qa(_pdf(), "type: resume",
                                describe_fn=self._describe(reply)))
        assert rep.ran is True and rep.ok is True
        assert rep.page_count == 1 and rep.issues == []
        assert "visually checked" in rep.note()

    def test_high_issue_fails(self):
        reply = json.dumps({"meets_rubric": False, "page_count": 2, "issues": [
            {"kind": "missing", "detail": "no skills section", "severity": "high"}]})
        rep = _run(vq.visual_qa(_pdf(), "must include skills",
                                describe_fn=self._describe(reply)))
        assert rep.ok is False and rep.blocking_issues

    def test_page_hint_mismatch_flags(self):
        reply = json.dumps({"meets_rubric": True, "page_count": 2, "issues": []})
        rep = _run(vq.visual_qa(_pdf(), "pages: exactly 1",
                                describe_fn=self._describe(reply), page_hint=1))
        assert rep.ok is False
        assert any(i.kind == "pages" for i in rep.issues)

    def test_empty_vlm_reply_fail_open(self):
        rep = _run(vq.visual_qa(_pdf(), "x", describe_fn=self._describe("")))
        assert rep.ran is False and rep.ok is True

    def test_vlm_error_fail_open(self):
        async def boom(images, prompt):
            raise RuntimeError("vlm down")
        rep = _run(vq.visual_qa(_pdf(), "x", describe_fn=boom))
        assert rep.ran is False and rep.ok is True

    def test_bad_pdf_skips(self):
        rep = _run(vq.visual_qa(b"garbage", "x",
                                describe_fn=self._describe("{}")))
        assert rep.ran is False and rep.note() == "rendered, re-opened"

    def test_malformed_json_fail_open(self):
        rep = _run(vq.visual_qa(_pdf(), "x",
                                describe_fn=self._describe("not json")))
        assert rep.ran is False and rep.ok is True
