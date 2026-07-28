"""Flow 2 — live local-first routing: with the on-pod floor enabled, live turns
pin the LOCAL model for a guaranteed fast first token; without it (dev boxes) or
with the flag off, behaviour is byte-identical to the configured live_model."""
from app.core.config_loader import cfg
from app.core.orchestrator import _live_model


def _set_local(monkeypatch, enabled, model_id="qwen2.5-14b-instruct"):
    import app.llm.catalog as cat
    monkeypatch.setattr(cat, "local_enabled", lambda: enabled)
    monkeypatch.setattr(cat, "local_model_id", lambda: model_id)


def test_pins_local_when_floor_enabled(monkeypatch):
    _set_local(monkeypatch, True)
    monkeypatch.setattr(cfg.routing, "live_local_first", True, raising=False)
    assert _live_model() == "qwen2.5-14b-instruct"


def test_flag_off_keeps_configured_pin(monkeypatch):
    _set_local(monkeypatch, True)
    monkeypatch.setattr(cfg.routing, "live_local_first", False, raising=False)
    monkeypatch.setattr(cfg.llm, "live_model", "llama-3.3-70b-versatile",
                        raising=False)
    assert _live_model() == "llama-3.3-70b-versatile"


def test_no_floor_is_a_noop(monkeypatch):
    _set_local(monkeypatch, False)
    monkeypatch.setattr(cfg.routing, "live_local_first", True, raising=False)
    monkeypatch.setattr(cfg.llm, "live_model", "llama-3.3-70b-versatile",
                        raising=False)
    assert _live_model() == "llama-3.3-70b-versatile"


def test_escalation_call_site_bypasses_pin():
    # The pin is applied as `None if ctx.escalate else _live_model()` at the
    # call site — assert that contract still exists so verify/escalation keeps
    # reaching the stronger cloud chain.
    import inspect
    from app.core import orchestrator
    src = inspect.getsource(orchestrator)
    assert "None if ctx.escalate else _live_model()" in src
