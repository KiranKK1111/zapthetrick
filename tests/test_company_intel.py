"""Stage-6 §4.12 — company intelligence: SSRF guard, OrgBrief, org-Q, cache."""
from __future__ import annotations

import asyncio

import pytest

from app.live import company_intel as CI


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh():
    CI.reset_for_tests()
    yield
    CI.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "company_intel", True, raising=False)


class TestSsrfGuard:
    def test_public_https_is_safe(self):
        assert CI.is_safe_url("https://example.com") is True

    @pytest.mark.parametrize("bad", [
        "http://localhost/x",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data",   # cloud metadata
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.9/",
        "https://api.internal/",
        "https://box.local/",
        "ftp://example.com/",
        "file:///etc/passwd",
        "",
    ])
    def test_unsafe_urls_blocked(self, bad):
        assert CI.is_safe_url(bad) is False

    def test_ipv6_loopback_blocked(self):
        assert CI.is_safe_url("http://[::1]/") is False


class TestOrgQuestion:
    def test_company_question_detected(self):
        assert CI.is_org_question("what does our company do") is True
        assert CI.is_org_question("why do you want to work here") is True

    def test_self_question_not_org(self):
        assert CI.is_org_question("tell me about yourself") is False
        assert CI.is_org_question("what are your strengths") is False

    def test_empty_is_false(self):
        assert CI.is_org_question("") is False


class TestOrgBrief:
    def test_directive_carries_facts(self):
        b = CI.OrgBrief(
            company="Acme", identity="Acme builds developer tools.",
            product="an API gateway", stack=["Go", "Kafka"],
            focus=["reliability"], trajectory={"recent": "Series B"})
        d = b.directive()
        assert "Acme" in d and "API gateway" in d
        assert "Go" in d and "Kafka" in d and "Series B" in d

    def test_empty_brief_directive_blank(self):
        assert CI.OrgBrief(company="X").directive() == ""

    def test_as_dict_shape(self):
        b = CI.OrgBrief(company="Acme", product="p")
        assert CI.OrgBrief(**{}, company="Acme").company == "Acme" \
            if False else b.as_dict()["company"] == "Acme"


class TestCache:
    def test_put_then_get(self):
        CI.cache_put("Acme", CI.OrgBrief(company="Acme", product="p"), now=1000)
        got = CI.cache_get("Acme", now=1000)
        assert got is not None and got.product == "p"

    def test_case_insensitive_key(self):
        CI.cache_put("Acme Corp", CI.OrgBrief(company="Acme Corp"), now=1000)
        assert CI.cache_get("acme corp", now=1000) is not None

    def test_expires_after_ttl(self):
        CI.cache_put("Acme", CI.OrgBrief(company="Acme"), now=1000)
        # 8 days later (> 7-day TTL) → stale.
        assert CI.cache_get("Acme", now=1000 + 8 * 86_400) is None

    def test_miss_returns_none(self):
        assert CI.cache_get("Nobody") is None


class TestBuildBrief:
    def test_disabled_returns_none(self, monkeypatch):
        # Force the flag OFF (the deployment default is now ON) so this exercises
        # the disabled path regardless of config.yaml.
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "company_intel", False, raising=False)
        assert _run(CI.build_brief("Acme", url="https://acme.com",
                                   crawl_fn=lambda u: {})) is None

    def test_crawls_and_assembles(self, _on):
        def crawl(url):
            return {"identity": "Acme builds tools.", "product": "an API",
                    "stack": ["Go"], "trajectory": {"recent": "Series B"}}
        b = _run(CI.build_brief("Acme", url="https://acme.com", crawl_fn=crawl))
        assert b is not None and b.product == "an API"
        assert b.source_url == "https://acme.com"
        # Cached now → a second call serves without crawling.
        def _boom(u):
            raise AssertionError("should not re-crawl")
        b2 = _run(CI.build_brief("Acme", url="https://acme.com", crawl_fn=_boom))
        assert b2 is not None and b2.product == "an API"

    def test_unsafe_url_refused(self, _on):
        called = {"n": 0}
        def crawl(u):
            called["n"] += 1
            return {}
        assert _run(CI.build_brief("Acme", url="http://169.254.169.254/",
                                   crawl_fn=crawl)) is None
        assert called["n"] == 0                       # crawl never invoked

    def test_async_crawl_fn(self, _on):
        async def crawl(url):
            return {"product": "an async API"}
        b = _run(CI.build_brief("Acme", url="https://acme.com", crawl_fn=crawl))
        assert b.product == "an async API"

    def test_no_url_returns_none(self, _on):
        assert _run(CI.build_brief("Acme")) is None

    def test_crawl_error_fail_open(self, _on):
        def crawl(u):
            raise RuntimeError("crawl exploded")
        assert _run(CI.build_brief("Acme", url="https://acme.com",
                                   crawl_fn=crawl)) is None
