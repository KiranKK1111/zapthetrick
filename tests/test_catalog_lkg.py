"""Last-known-good catalog invariant (vNext §2.8).

A provider's failed /models fetch must NEVER delete or mutate the existing
catalog — the pool is never emptied by a bad day. This pins that discovery
returns before ever opening a DB session on any fetch failure.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from app.llm import discovery as D


def _install_failing_fetch(monkeypatch, *, raise_transport=True, status=200,
                           body_ok=True):
    # A real-enough provider spec (non-Cloudflare bearer provider).
    spec = SimpleNamespace(auth="bearer", base_url="https://prov.test/v1",
                           extra_headers={}, allow_anonymous=False)
    monkeypatch.setattr(D, "get_provider_spec", lambda p: spec)

    # A session factory that FAILS LOUDLY if discovery ever opens a session on a
    # fetch-failure path (i.e. reaches the DB write/delete block).
    opened = {"n": 0}

    class _Factory:
        def __call__(self):
            opened["n"] += 1
            raise AssertionError("DB session opened despite a fetch failure — "
                                 "the LKG invariant is broken")

    monkeypatch.setattr(D, "get_session_factory", lambda: _Factory())

    class _FakeResp:
        status_code = status

        def json(self):
            if not body_ok:
                raise ValueError("non-JSON")
            return {"data": [{"id": "some-model"}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            if raise_transport:
                raise httpx.HTTPError("connection refused")
            return _FakeResp()

    monkeypatch.setattr(D.httpx, "AsyncClient", _FakeClient)
    return opened


def test_transport_error_never_opens_a_db_session(monkeypatch):
    opened = _install_failing_fetch(monkeypatch, raise_transport=True)
    res = asyncio.run(D.discover_models("prov", api_key="x"))
    assert res["added"] == 0
    assert "error" in res
    assert opened["n"] == 0            # catalog untouched


def test_non_200_never_opens_a_db_session(monkeypatch):
    opened = _install_failing_fetch(monkeypatch, raise_transport=False, status=503)
    res = asyncio.run(D.discover_models("prov", api_key="x"))
    assert res["added"] == 0
    assert "503" in res["error"]
    assert opened["n"] == 0


def test_non_json_never_opens_a_db_session(monkeypatch):
    opened = _install_failing_fetch(monkeypatch, raise_transport=False,
                                    status=200, body_ok=False)
    res = asyncio.run(D.discover_models("prov", api_key="x"))
    assert res["added"] == 0
    assert "error" in res
    assert opened["n"] == 0
