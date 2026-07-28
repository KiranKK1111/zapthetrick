"""Gap 5 — AEC seam: no-op passthrough by default; native processor slots in."""
from app.core.config_loader import cfg
from app.live import aec


def test_default_is_passthrough():
    aec._registered = None
    assert aec.process([1, 2, 3]) == [1, 2, 3]        # mic unchanged
    assert isinstance(aec.get_aec(), aec._NoopAec)


def test_registered_native_used_when_enabled(monkeypatch):
    class FakeAec:
        def process(self, mic, reference=None):
            return "cancelled"
    aec.register_aec(FakeAec())
    monkeypatch.setattr(cfg.voice, "native_aec", True, raising=False)
    assert aec.process([1, 2]) == "cancelled"
    # disabled → passthrough even if a native one is registered.
    monkeypatch.setattr(cfg.voice, "native_aec", False, raising=False)
    assert aec.process([1, 2]) == [1, 2]
    aec._registered = None


def test_fail_open(monkeypatch):
    class BoomAec:
        def process(self, mic, reference=None):
            raise RuntimeError("boom")
    aec.register_aec(BoomAec())
    monkeypatch.setattr(cfg.voice, "native_aec", True, raising=False)
    assert aec.process([9]) == [9]                     # error → mic unchanged
    aec._registered = None
