"""Stage-4 §3.4 Component C — provider pre-connect + stream-resume marker.

Pre-connect warms HTTP/2 pools to the top-N candidate providers so a turn (and
its hedge) fires on an already-open connection; the continuation wrapper now
surfaces a recovered DROPPED stream as 'stream_resumed' in an optional degraded
sink. Both additive + fail-open (off by default). Idempotency + continuation
themselves are already covered by test_idempotency.py / test_continuation.py.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from app.llm import engine, preconnect


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    preconnect.reset_for_tests()
    yield
    preconnect.reset_for_tests()


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.heads: list[str] = []
        self.fail = fail

    async def head(self, url, timeout=None):
        self.heads.append(url)
        if self.fail:
            raise RuntimeError("boom")
        return types.SimpleNamespace(status_code=405)


def _pre_on(monkeypatch, top_n=3, interval=30.0):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.resilience, "pre_connect", True, raising=False)
    monkeypatch.setattr(cfg.resilience, "pre_connect_top_n", top_n, raising=False)
    monkeypatch.setattr(cfg.resilience, "pre_connect_min_interval_s", interval,
                        raising=False)


def _stub_keys(monkeypatch, platforms):
    async def fake_list_keys(platform=None):
        return [types.SimpleNamespace(platform=p, enabled=True)
                for p in platforms]
    from app.llm import keys
    monkeypatch.setattr(keys, "list_keys", fake_list_keys)


# --------------------------------------------------------------------------- #
class TestHostBase:
    def test_strips_path_to_scheme_host(self):
        assert (preconnect._host_base("https://api.groq.com/openai/v1")
                == "https://api.groq.com")

    def test_skips_templated_host(self):
        url = ("https://api.cloudflare.com/client/v4/accounts/"
               "{account_id}/ai/v1")
        assert preconnect._host_base(url) is None

    def test_bad_input_is_none(self):
        assert preconnect._host_base("") is None
        assert preconnect._host_base("not-a-url") is None


class TestCandidateBases:
    def test_configured_first_deduped_capped(self, monkeypatch):
        _stub_keys(monkeypatch, ["groq", "cerebras"])
        bases = _run(preconnect._candidate_bases(2))
        assert bases == ["https://api.groq.com", "https://api.cerebras.ai"]

    def test_dedups_same_platform(self, monkeypatch):
        _stub_keys(monkeypatch, ["groq", "groq"])
        bases = _run(preconnect._candidate_bases(5))
        assert bases.count("https://api.groq.com") == 1

    def test_no_keys_uses_anonymous_tier(self, monkeypatch):
        _stub_keys(monkeypatch, [])
        bases = _run(preconnect._candidate_bases(3))
        # Registry has anonymous-tier providers; a list (possibly some hosts).
        assert isinstance(bases, list)


class TestWarm:
    def test_warms_each_candidate_host(self, monkeypatch):
        _pre_on(monkeypatch, top_n=2)
        _stub_keys(monkeypatch, ["groq", "cerebras"])
        fc = _FakeClient()
        import app.core.http_pool as hp
        monkeypatch.setattr(hp, "get_http_client", lambda: fc)
        n = _run(preconnect.warm())
        assert n == 2
        assert set(fc.heads) == {"https://api.groq.com",
                                 "https://api.cerebras.ai"}

    def test_off_returns_zero(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.resilience, "pre_connect", False, raising=False)
        assert _run(preconnect.warm()) == 0

    def test_force_warms_when_off(self, monkeypatch):
        # §3.10 input-warmup drives warming on its own flag (pre_connect off).
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.resilience, "pre_connect", False, raising=False)
        _stub_keys(monkeypatch, ["groq"])
        fc = _FakeClient()
        import app.core.http_pool as hp
        monkeypatch.setattr(hp, "get_http_client", lambda: fc)
        assert _run(preconnect.warm(force=True, top_n=1)) == 1

    def test_fail_open_on_client_error(self, monkeypatch):
        _pre_on(monkeypatch, top_n=1)
        _stub_keys(monkeypatch, ["groq"])
        fc = _FakeClient(fail=True)
        import app.core.http_pool as hp
        monkeypatch.setattr(hp, "get_http_client", lambda: fc)
        n = _run(preconnect.warm())          # never raises
        assert n == 0
        assert fc.heads == ["https://api.groq.com"]  # attempted (still warms TLS)

    def test_top_n_caps_hosts(self, monkeypatch):
        _pre_on(monkeypatch, top_n=1)
        _stub_keys(monkeypatch, ["groq", "cerebras", "mistral"])
        fc = _FakeClient()
        import app.core.http_pool as hp
        monkeypatch.setattr(hp, "get_http_client", lambda: fc)
        _run(preconnect.warm())
        assert len(fc.heads) == 1


class TestSchedule:
    def test_off_is_noop(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.resilience, "pre_connect", False, raising=False)

        async def go():
            called: list[int] = []

            async def fake_warm(**kw):
                called.append(1)
            monkeypatch.setattr(preconnect, "warm", fake_warm)
            preconnect.schedule()
            await asyncio.sleep(0.01)
            return called
        assert _run(go()) == []

    def test_debounced_within_interval(self, monkeypatch):
        _pre_on(monkeypatch, interval=999.0)

        async def go():
            called: list[int] = []

            async def fake_warm(**kw):
                called.append(1)
            monkeypatch.setattr(preconnect, "warm", fake_warm)
            preconnect.schedule()
            preconnect.schedule()          # within the interval → skipped
            await asyncio.sleep(0.02)
            return called
        assert len(_run(go())) == 1


# --------------------------------------------------------------------------- #
# stream-resume: 'stream_resumed' surfaces a recovered dropped stream
# --------------------------------------------------------------------------- #
class TestStreamResumedMarker:
    def _no_cutoff(self, monkeypatch):
        # Isolate from any global usage state so a clean end never auto-continues.
        import app.llm.usage as _usage
        monkeypatch.setattr(_usage, "finish_reason", lambda *a, **k: None,
                            raising=False)

    def test_marker_appended_on_mid_stream_recovery(self, monkeypatch):
        self._no_cutoff(monkeypatch)
        calls = {"n": 0}

        async def fake_ras(msgs, opts, *, session_key=None,
                           preferred_model_db_id=None):
            calls["n"] += 1
            if calls["n"] == 1:
                yield "hello"
                raise RuntimeError("mid-stream drop")
            else:
                yield " world"
        monkeypatch.setattr(engine, "route_and_stream", fake_ras)

        sink: list[str] = []

        async def go():
            out = []
            async for c in engine.stream_with_continuation(
                    [{"role": "user", "content": "hi"}],
                    {"mid_stream_continuation": True, "_degraded_sink": sink}):
                out.append(c)
            return "".join(out)
        text = _run(go())
        assert "stream_resumed" in sink
        assert "hello" in text and "world" in text

    def test_no_marker_on_clean_stream(self, monkeypatch):
        self._no_cutoff(monkeypatch)

        async def fake_ras(msgs, opts, *, session_key=None,
                           preferred_model_db_id=None):
            yield "all good"
        monkeypatch.setattr(engine, "route_and_stream", fake_ras)

        sink: list[str] = []

        async def go():
            return [c async for c in engine.stream_with_continuation(
                    [{"role": "user", "content": "hi"}],
                    {"mid_stream_continuation": True, "_degraded_sink": sink})]
        _run(go())
        assert sink == []

    def test_sink_key_never_reaches_provider(self, monkeypatch):
        self._no_cutoff(monkeypatch)
        seen_opts: list[dict] = []

        async def fake_ras(msgs, opts, *, session_key=None,
                           preferred_model_db_id=None):
            seen_opts.append(opts)
            yield "ok"
        monkeypatch.setattr(engine, "route_and_stream", fake_ras)

        async def go():
            return [c async for c in engine.stream_with_continuation(
                    [{"role": "user", "content": "hi"}],
                    {"mid_stream_continuation": True, "_degraded_sink": []})]
        _run(go())
        assert all("_degraded_sink" not in o for o in seen_opts)
