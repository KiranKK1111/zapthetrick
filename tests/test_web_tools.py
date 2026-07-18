"""P2-6 — in-loop web tools (WebSearch / WebFetch parity).

Pure/offline: the SSRF guard, HTML→text, the search/fetch tools (network
monkeypatched), the untrusted-content banner, config gating, and the loop
exposing/hiding the tools. No real network.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent import tools
from app.agent.webtools import check_url, html_to_text, web_fetch, web_search


# ── SSRF guard ──────────────────────────────────────────────────────────────
def test_check_url_rejects_non_http_and_internal():
    assert not check_url("ftp://example.com")[0]
    assert not check_url("file:///etc/passwd")[0]
    assert not check_url("http://localhost/x")[0]
    assert not check_url("http://127.0.0.1/x")[0]
    assert not check_url("http://169.254.169.254/latest/meta-data")[0]
    assert not check_url("http://10.0.0.5/")[0]
    assert not check_url("http://service.internal/")[0]
    assert not check_url("notaurl")[0]


def test_check_url_allows_public(monkeypatch):
    # Force resolution to a public IP so the test doesn't hit DNS.
    import app.agent.webtools as wt
    monkeypatch.setattr(wt, "_is_public_host", lambda host: True)
    ok, reason = check_url("https://docs.python.org/3/")
    assert ok and reason == ""


# ── HTML → text ───────────────────────────────────────────────────────────────
def test_html_to_text_strips_tags_and_scripts():
    html = ("<html><head><style>.x{}</style><script>evil()</script></head>"
            "<body><h1>Title</h1><p>Hello &amp; welcome</p></body></html>")
    text = html_to_text(html)
    assert "Title" in text and "Hello & welcome" in text
    assert "evil()" not in text and "<p>" not in text


# ── web_search (network monkeypatched) ───────────────────────────────────────
def test_web_search_formats_results(monkeypatch):
    async def fake_search(*, query, max_results=5):
        assert query == "graphql federation"
        return [
            {"title": "Fed v2", "url": "https://x.dev/fed", "snippet": "about"},
            {"title": "Guide", "url": "https://y.dev/g", "snippet": "more"},
        ]
    monkeypatch.setattr("app.tools.web_search.search", fake_search)
    out = asyncio.run(web_search("graphql federation"))
    assert "UNTRUSTED CONTENT" in out
    assert "Fed v2" in out and "https://x.dev/fed" in out


def test_web_search_empty_query():
    assert asyncio.run(web_search("  ")).startswith("ERROR")


def test_web_search_handles_failure(monkeypatch):
    async def boom(*, query, max_results=5):
        raise RuntimeError("ddg down")
    monkeypatch.setattr("app.tools.web_search.search", boom)
    out = asyncio.run(web_search("x"))
    assert out.startswith("ERROR") and "ddg down" in out


# ── web_fetch (network + DNS monkeypatched) ──────────────────────────────────
class _Resp:
    def __init__(self, text, status=200, ctype="text/html",
                 url="https://example.com/"):
        self.content = text.encode("utf-8")
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.encoding = "utf-8"
        self.url = url


def _patch_httpx(monkeypatch, resp):
    import app.agent.webtools as wt
    monkeypatch.setattr(wt, "_is_public_host", lambda host: True)

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return resp

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def test_web_fetch_returns_readable_text(monkeypatch):
    _patch_httpx(monkeypatch, _Resp(
        "<html><body><h1>Docs</h1><p>Install with pip.</p></body></html>"))
    out = asyncio.run(web_fetch("https://example.com/docs"))
    assert "UNTRUSTED CONTENT" in out
    assert "Docs" in out and "Install with pip." in out


def test_web_fetch_blocks_internal_without_network():
    out = asyncio.run(web_fetch("http://localhost:8000/admin"))
    assert out.startswith("ERROR")


def test_web_fetch_http_error(monkeypatch):
    _patch_httpx(monkeypatch, _Resp("nope", status=404))
    out = asyncio.run(web_fetch("https://example.com/missing"))
    assert "HTTP 404" in out


def test_web_fetch_blocks_redirect_to_internal(monkeypatch):
    import app.agent.webtools as wt
    # public on the first check, internal on the final-URL check
    seq = iter([True, False])
    monkeypatch.setattr(wt, "_is_public_host",
                        lambda host: next(seq, False))
    resp = _Resp("<html>x</html>", url="http://10.0.0.9/internal")

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return resp
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    out = asyncio.run(web_fetch("https://example.com/redir"))
    assert "blocked host" in out


# ── tool registry + config gating ─────────────────────────────────────────────
def test_web_tools_registered():
    assert "web_search" in tools.HANDLERS and "web_fetch" in tools.HANDLERS
    assert "web_search" in tools.SPEC_BY_NAME


def test_tools_doc_can_exclude_web():
    doc = tools.tools_doc(exclude={"web_search", "web_fetch"})
    assert "web_search" not in doc and "web_fetch" not in doc
    full = tools.tools_doc()
    assert "web_search" in full


def test_web_tool_handler_respects_disable(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.advanced_rag, "agent_web_tools", False)
    out = asyncio.run(tools.web_search("/ws", query="x"))
    assert "disabled" in out


def test_web_tools_runs_flag_blocks_plan_mode():
    # plan mode is read-only; web tools (runs=True) must be denied there.
    from app.agent import permissions
    d, _ = permissions.decide("web_fetch", {"url": "https://x.dev"}, "plan")
    assert d == "deny"
    d2, _ = permissions.decide("web_fetch", {"url": "https://x.dev"},
                               "acceptEdits")
    assert d2 == "allow"
