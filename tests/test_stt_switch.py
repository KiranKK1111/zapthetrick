"""Tracked STT engine switching (2026-07-12): the observable state machine
behind the Settings popup — unload metrics, download sampling, warm-verified
ready state, single-flight, endpoint contracts, and the VPS config
preservation that makes the selection survive redeploys."""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from starlette.testclient import TestClient

from app.stt import switch


def _fresh_state():
    switch._STATE.clear()
    switch._STATE["phase"] = "idle"


class TestSwitchStateMachine:
    def setup_method(self):
        _fresh_state()

    def _stub_factory(self, monkeypatch, *, resident_before, resident_after,
                      warm_error=None):
        from app.stt import factory
        calls = {"unload": 0, "warm": 0}
        seq = {"resident": list(resident_before)}
        monkeypatch.setattr(factory, "resident_engines",
                            lambda: list(seq["resident"]))

        def unload():
            calls["unload"] += 1
            seq["resident"] = []
        monkeypatch.setattr(factory, "unload_all", unload)

        async def warm():
            calls["warm"] += 1
            if warm_error:
                raise warm_error
            seq["resident"] = list(resident_after)
        monkeypatch.setattr(factory, "warm_active", warm)
        return calls

    def test_happy_path_reaches_ready(self, monkeypatch):
        calls = self._stub_factory(monkeypatch,
                                   resident_before=["parakeet"],
                                   resident_after=["qwen_asr"])
        monkeypatch.setattr(switch, "_dir_bytes", lambda p: 42_000_000)
        asyncio.run(switch._run("qwen_asr"))
        s = switch.state()
        assert s["phase"] == "ready"
        assert s["freed_engines"] == ["parakeet"]
        assert s["downloaded_bytes"] == 42_000_000
        assert calls == {"unload": 1, "warm": 1}

    def test_engine_not_resident_after_warm_is_error(self, monkeypatch):
        self._stub_factory(monkeypatch, resident_before=[],
                           resident_after=[])   # warm "succeeds" but no load
        monkeypatch.setattr(switch, "_dir_bytes", lambda p: 0)
        asyncio.run(switch._run("parakeet"))
        s = switch.state()
        assert s["phase"] == "error"
        assert "retry" in (s["error"] or "")

    def test_warm_exception_is_error_not_crash(self, monkeypatch):
        self._stub_factory(monkeypatch, resident_before=[],
                           resident_after=[],
                           warm_error=RuntimeError("download failed"))
        monkeypatch.setattr(switch, "_dir_bytes", lambda p: 0)
        asyncio.run(switch._run("qwen_asr"))
        s = switch.state()
        assert s["phase"] == "error"
        assert "download failed" in s["error"]

    def test_already_downloaded_goes_straight_to_loading(self, monkeypatch):
        phases: list[str] = []
        from app.stt import factory
        monkeypatch.setattr(factory, "resident_engines", lambda: ["parakeet"])
        monkeypatch.setattr(factory, "unload_all", lambda: None)

        async def warm():
            phases.append(switch._STATE["phase"])
        monkeypatch.setattr(factory, "warm_active", warm)
        monkeypatch.setattr(switch, "_dir_bytes", lambda p: 500_000_000)
        asyncio.run(switch._run("parakeet"))
        assert phases == ["loading"]           # skipped "downloading"
        assert switch.state()["was_downloaded"] is True

    def test_fresh_model_shows_downloading(self, monkeypatch):
        phases: list[str] = []
        from app.stt import factory
        monkeypatch.setattr(factory, "resident_engines",
                            lambda: ["qwen_asr"])
        monkeypatch.setattr(factory, "unload_all", lambda: None)

        async def warm():
            phases.append(switch._STATE["phase"])
        monkeypatch.setattr(factory, "warm_active", warm)
        monkeypatch.setattr(switch, "_dir_bytes", lambda p: 0)
        asyncio.run(switch._run("qwen_asr"))
        assert phases == ["downloading"]
        assert switch.state()["was_downloaded"] is False

    def test_single_flight_joins_same_target(self, monkeypatch):
        runs = {"n": 0}

        async def fake_run(target):
            runs["n"] += 1
            switch._STATE["to"] = target
            await asyncio.sleep(0.05)
        monkeypatch.setattr(switch, "_run", fake_run)

        async def go():
            await switch.start_switch("qwen_asr")
            await asyncio.sleep(0.01)
            await switch.start_switch("qwen_asr")   # joins, no second run
            await asyncio.sleep(0.1)
        asyncio.run(go())
        assert runs["n"] == 1

    def test_new_target_supersedes(self, monkeypatch):
        order: list[str] = []

        async def fake_run(target):
            switch._STATE["to"] = target
            order.append(target)
            await asyncio.sleep(0.05)
        monkeypatch.setattr(switch, "_run", fake_run)

        async def go():
            await switch.start_switch("qwen_asr")
            await asyncio.sleep(0.01)
            await switch.start_switch("parakeet")
            await asyncio.sleep(0.1)
        asyncio.run(go())
        assert order == ["qwen_asr", "parakeet"]

    def test_hf_dir_mapping(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.stt, "model", "small.en", raising=False)
        p1 = switch._hf_dir_for("parakeet")
        p2 = switch._hf_dir_for("qwen_asr")
        p3 = switch._hf_dir_for("faster_whisper")
        assert "istupakov" in str(p1)
        assert "Qwen" in str(p2)
        assert "faster-whisper-small.en" in str(p3)
        assert switch._hf_dir_for("unknown") is None


class TestSttEndpoints:
    def _app(self) -> FastAPI:
        from app.api.routes_stt import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_select_rejects_unknown(self):
        c = TestClient(self._app())
        r = c.post("/api/stt/select", json={"id": "groq::whisper-large-v3"})
        assert r.status_code == 400
        r2 = c.post("/api/stt/select", json={"id": "nonsense"})
        assert r2.status_code == 400

    def test_select_persists_exclusive_settings(self, monkeypatch):
        captured: dict = {}

        async def fake_write(updates):
            captured.update(updates)
            return updates
        import app.api.routes_settings as rset
        monkeypatch.setattr(rset, "write_settings", fake_write)

        async def fake_start(target):
            captured["switch_target"] = target
        monkeypatch.setattr(switch, "start_switch", fake_start)

        c = TestClient(self._app())
        r = c.post("/api/stt/select",
                   json={"id": "faster_whisper::small.en"})
        assert r.status_code == 200
        stt = captured["stt"]
        assert stt["provider"] == "faster_whisper"
        assert stt["model"] == "small.en"
        assert stt["partial_provider"] == "faster_whisper"
        assert stt["fallback_providers"] == []
        assert stt["dual_engine_enabled"] is False
        assert captured["switch_target"] == "faster_whisper::small.en"

    def test_status_shape(self, monkeypatch):
        _fresh_state()
        switch._STATE.update({"phase": "ready", "to": "parakeet",
                              "freed_engines": ["qwen_asr"],
                              "downloaded_bytes": 123, "error": None})
        from app.stt import factory
        monkeypatch.setattr(factory, "resident_engines",
                            lambda: ["parakeet"])
        c = TestClient(self._app())
        body = c.get("/api/stt/status").json()
        assert body["phase"] == "ready"
        assert body["freed_engines"] == ["qwen_asr"]
        assert body["resident_engines"] == ["parakeet"]
        assert "active" in body


# NOTE: TestVpsConfigPreservation was removed with deploy/render_config.py — the
# Hostinger/VPS Docker deploy is gone (the pod now writes config in
# deploy/runpod_entrypoint.sh). STT selection still persists on /workspace.
