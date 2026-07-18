"""Phase 3 — Document quality analyzer (deterministic reviewer + gate)."""
from __future__ import annotations

import asyncio

from app.documents.planner import plan_blueprint, DocGoal, Depth
from app.documents.review import (
    QualityReport, analyze_document, analyze_document_async,
    code_lint_document, llm_review_document, multi_reviewer_document, safe_fix,
)


def _cats(report):
    return {i.category for i in report.issues}


def _patch_docs_cfg(monkeypatch, **flags):
    """Override cfg.documents flags that aren't real Pydantic fields (the new
    doc flags run on getattr defaults; config_loader is owned elsewhere). Swaps
    a lightweight namespace in for the whole cfg so the lazy
    ``from app.core.config_loader import cfg`` inside the reviewers sees it."""
    import types

    from app.core import config_loader
    # Every reader uses getattr(cfg.documents, flag, <default>), so a namespace
    # carrying only the overridden flags is enough — the rest hit their defaults.
    ns = types.SimpleNamespace(**flags)
    monkeypatch.setattr(config_loader, "cfg",
                        types.SimpleNamespace(documents=ns))


class TestLLMReviewPanel:
    """The flag-gated LLM reviewer is additive + fail-open (default OFF)."""

    _MD = "# Title\n\nintro paragraph.\n\n## Section\n\nreal content.\n"

    def test_llm_review_disabled_by_default_is_empty(self):
        assert asyncio.run(llm_review_document(self._MD)) == []

    def test_async_matches_sync_when_llm_off(self):
        sync = analyze_document(self._MD)
        asy = asyncio.run(analyze_document_async(self._MD))
        assert asy.score == sync.score
        assert _cats(asy) == _cats(sync)

    def test_llm_panel_merges_and_rescores_when_on(self, monkeypatch):
        import app.documents.review as rv

        async def _fake_complete(messages, model=None, options=None):
            return ('[{"severity":"warning","category":"clarity",'
                    '"message":"Ambiguous sentence in intro.","section":"Title"}]')

        class _Cfg:
            class documents:
                llm_review = True
        monkeypatch.setattr(rv, "cfg", _Cfg, raising=False)
        # patch the module the reviewer imports lazily
        from app.core import llm_client
        monkeypatch.setattr(llm_client.llm, "complete", _fake_complete)
        # force the flag read to see llm_review=True
        from app.core import config_loader
        monkeypatch.setattr(config_loader.cfg.documents, "llm_review", True,
                            raising=False)
        issues = asyncio.run(llm_review_document(self._MD))
        assert any(i.category == "clarity" for i in issues)

    def test_llm_panel_failopen_on_bad_json(self, monkeypatch):
        async def _bad(messages, model=None, options=None):
            return "not json at all"
        from app.core import llm_client, config_loader
        monkeypatch.setattr(config_loader.cfg.documents, "llm_review", True,
                            raising=False)
        monkeypatch.setattr(llm_client.llm, "complete", _bad)
        assert asyncio.run(llm_review_document(self._MD)) == []


class TestCodeLintPass:
    """Phase 3 — wire polyglot/linters into review. Deterministic mapping is
    proven by faking the linter; the real path is fail-open when no binary."""

    _MD = ("# Script\n\n```python\nimport os\nx=1\n```\n")

    def test_no_lintable_code_is_empty(self):
        assert asyncio.run(code_lint_document("# T\n\njust prose.")) == []

    def test_maps_findings_to_issues(self, monkeypatch):
        import app.polyglot.linters as linters

        async def _fake_lint(lang, code, **kw):
            return [linters.LintFinding(line=2, message="unused import 'os'",
                                        code="F401", severity="warning")]

        monkeypatch.setattr(linters, "lint_code", _fake_lint)
        # code_lint_review defaults ON — no flag flip needed.
        issues = asyncio.run(code_lint_document(self._MD))
        assert issues and issues[0].category == "code_lint"
        assert "F401" in issues[0].message and "unused import" in issues[0].message

    def test_disabled_by_flag(self, monkeypatch):
        _patch_docs_cfg(monkeypatch, code_lint_review=False)
        assert asyncio.run(code_lint_document(self._MD)) == []

    def test_failopen_when_linter_raises(self, monkeypatch):
        import app.polyglot.linters as linters

        async def _boom(*a, **kw):
            raise RuntimeError("linter exploded")

        monkeypatch.setattr(linters, "lint_code", _boom)
        assert asyncio.run(code_lint_document(self._MD)) == []


class TestMultiReviewerPanel:
    """Phase 3 — role-split reviewers (LLM, flag-gated, fail-open)."""

    _MD = "# Title\n\nSome content to review.\n"

    def test_disabled_by_default(self):
        assert asyncio.run(multi_reviewer_document(self._MD)) == []

    def test_roles_run_and_tag_category(self, monkeypatch):
        from app.core import llm_client

        async def _fake(messages, model=None, options=None):
            return ('[{"severity":"warning","message":"issue found",'
                    '"section":""}]')

        _patch_docs_cfg(monkeypatch, multi_reviewer=True)
        monkeypatch.setattr(llm_client.llm, "complete", _fake)
        issues = asyncio.run(multi_reviewer_document(self._MD))
        cats = {i.category for i in issues}
        # one issue per role, each tagged review_<role>
        assert {"review_technical", "review_grammar", "review_formatting",
                "review_consistency"} <= cats

    def test_failopen_on_role_error(self, monkeypatch):
        from app.core import llm_client

        async def _boom(messages, model=None, options=None):
            raise RuntimeError("no route")

        _patch_docs_cfg(monkeypatch, multi_reviewer=True)
        monkeypatch.setattr(llm_client.llm, "complete", _boom)
        assert asyncio.run(multi_reviewer_document(self._MD)) == []


class TestSafeFix:
    """Phase 3 — bounded deterministic safe-fix (no LLM)."""

    def test_removes_consecutive_duplicate_paragraph(self):
        md = "# T\n\nHello world.\n\nHello world.\n\n## S\n\nOther.\n"
        fixed, applied = safe_fix(md)
        assert fixed.count("Hello world.") == 1
        assert any("duplicate" in a for a in applied)

    def test_collapses_internal_whitespace(self):
        md = "# T\n\nToo    many     spaces.\n"
        fixed, applied = safe_fix(md)
        assert "Too many spaces." in fixed
        assert applied

    def test_clean_document_reports_no_fixes(self):
        md = "# T\n\nA clean paragraph.\n\n## S\n\nMore content.\n"
        fixed, applied = safe_fix(md)
        assert applied == []

    def test_failopen_returns_original_on_bad_input(self):
        # A non-string, non-model input path still returns safely.
        fixed, applied = safe_fix(None)
        assert applied == []


class TestAsyncMergesCodeLint:
    def test_async_report_includes_code_lint(self, monkeypatch):
        import app.polyglot.linters as linters

        async def _fake_lint(lang, code, **kw):
            return [linters.LintFinding(line=1, message="bad", code="E1")]

        monkeypatch.setattr(linters, "lint_code", _fake_lint)
        md = "# T\n\nintro.\n\n## Code\n\n```python\nx=1\n```\n"
        report = asyncio.run(analyze_document_async(md))
        assert any(i.category == "code_lint" for i in report.issues)


class TestChecks:
    def test_clean_document_scores_100(self):
        md = ("# Title\n\nA clear intro paragraph.\n\n"
              "## Section\n\nWith real content here.\n")
        r = analyze_document(md)
        assert r.score == 100 and r.passed and r.issues == []

    def test_empty_section_is_an_error(self):
        md = "# Title\n\nintro\n\n## Empty\n\n## Next\n\nbody\n"
        r = analyze_document(md)
        assert "empty_section" in _cats(r)
        assert r.has_errors and not r.passed
        assert r.score < 100

    def test_heading_hierarchy_skip(self):
        md = "# Top\n\nbody\n\n### Deep\n\nbody\n"   # H1 → H3
        assert "heading_hierarchy" in _cats(analyze_document(md))

    def test_placeholder_flagged(self):
        md = "# T\n\nThis section is TODO and needs work.\n"
        assert "placeholder" in _cats(analyze_document(md))

    def test_duplicate_content(self):
        para = "This is a substantial paragraph repeated verbatim twice over.\n"
        md = f"# T\n\n{para}\n## S\n\n{para}"
        assert "duplicate_content" in _cats(analyze_document(md))

    def test_malformed_table(self):
        md = ("# T\n\n| A | B | C |\n|---|---|---|\n| 1 | 2 |\n")  # short row
        assert "malformed_table" in _cats(analyze_document(md))


class TestCompleteness:
    def test_missing_required_section_vs_blueprint(self):
        bp = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.MEDIUM)
        md = "# Overview\n\nintro\n\n## Architecture\n\nstuff\n"  # missing Implementation
        r = analyze_document(md, blueprint=bp)
        assert "missing_section" in _cats(r)
        assert any("Implementation" in i.message for i in r.issues)

    def test_all_required_present_no_missing(self):
        bp = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.QUICK)
        parts = [f"## {s.title}\n\ncontent for {s.title}\n" for s in bp.sections]
        md = "# Design\n\nintro\n\n" + "\n".join(parts)
        r = analyze_document(md, blueprint=bp)
        assert "missing_section" not in _cats(r)


class TestReportShape:
    def test_as_dict_and_score_clamped(self):
        r = analyze_document("# A\n\n## E1\n\n## E2\n\n## E3\n\n## E4\n\n## E5\n"
                             "\n## E6\n\n## E7\n")  # many empty sections
        assert r.score == 0                       # clamped, not negative
        js = r.as_dict()
        assert set(js) == {"score", "passed", "accessible", "confidence",
                           "issues"}
        assert js["passed"] is False and isinstance(js["issues"], list)

    def test_accepts_a_model_directly(self):
        from app.documents.model import markdown_to_model
        m = markdown_to_model("# T\n\nbody\n")
        assert isinstance(analyze_document(m), QualityReport)

    def test_fail_open_on_garbage(self):
        # Never raises; worst case an empty pass.
        assert analyze_document(None).passed
        assert analyze_document("").passed


class TestAccessibility:
    def test_image_without_alt_flagged(self):
        r = analyze_document("# T\n\nintro text here that is long.\n\n"
                             "![](chart.png)\n")
        assert "accessibility" in _cats(r) and r.accessible is False

    def test_image_with_alt_ok(self):
        r = analyze_document("# T\n\nintro text here that is long.\n\n"
                             "![a sales chart](chart.png)\n")
        assert r.accessible is True

    def test_accessible_in_report_dict(self):
        js = analyze_document("# T\n\nplenty of clear content here.\n").as_dict()
        assert "accessible" in js and "confidence" in js


class TestRouteThreadsTheBlueprint:
    """BUG (2026-07-14): the chat route PLANNED a blueprint for an
    ANSWER_AND_ARTIFACT turn and then called analyze_document WITHOUT it, so
    `_check_completeness` never ran against the plan — a document missing a
    required section shipped silently. The route's review helper now threads the
    planner's Blueprint into the analyzer."""

    _MD = "# Overview\n\nintro\n\n## Architecture\n\nstuff\n"  # no Implementation

    def test_blueprint_reaches_the_completeness_check(self):
        from app.api.routes_agents import _review_quality

        bp = plan_blueprint(DocGoal.TECHNICAL_DESIGN, Depth.MEDIUM)
        q = _review_quality(self._MD, bp)
        cats = {i["category"] for i in q["issues"]}
        assert "missing_section" in cats
        assert any("Implementation" in i["message"] for i in q["issues"])

    def test_without_a_blueprint_completeness_is_not_checked(self):
        # Every other intent (no blueprint planned) keeps today's behavior.
        from app.api.routes_agents import _review_quality

        q = _review_quality(self._MD, None)
        assert "missing_section" not in {i["category"] for i in q["issues"]}

    def test_review_helper_is_fail_open(self):
        from app.api.routes_agents import _review_quality

        assert _review_quality("", None)["passed"] is True
        # A blueprint-shaped object that explodes must not break the turn.
        class _Boom:
            @property
            def sections(self):
                raise RuntimeError("bad blueprint")
        assert _review_quality("# T\n\nbody\n", _Boom()) is None


class TestSectionConfidence:
    def test_hedging_and_placeholder_lower_confidence(self):
        md = ("# Doc\n\n## Weak\n\nWe might possibly do this, TODO decide.\n\n"
              "## Strong\n\nThis section states concrete facts with clear, "
              "specific, well-supported detail and no hedging at all here.\n")
        r = analyze_document(md)
        assert r.confidence["Weak"] < 0.6 < r.confidence["Strong"]
        assert "low_confidence" in _cats(r)

    def test_structural_section_not_flagged(self):
        # A title with only subsections (no direct prose) isn't scored/flagged.
        r = analyze_document("# Title\n\n## Real\n\nlots of solid content here "
                             "that is clearly written and detailed enough.\n")
        assert "Title" not in r.confidence     # no direct prose → skipped
        assert "low_confidence" not in _cats(r)
