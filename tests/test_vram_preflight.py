"""Gap 4 — VRAM preflight budget estimation."""
from app.core import vram_preflight as V
from app.core.config_loader import cfg


def _set_local(monkeypatch, model_id, small=""):
    class L:
        enabled = True
        model_id = ""
        small_model_id = ""
    L.model_id = model_id
    L.small_model_id = small
    monkeypatch.setattr(cfg.llm, "local", L, raising=False)


def test_params_parsed_from_id():
    assert V._params_b_from_id("qwen2.5-14b-instruct") == 14.0
    assert V._params_b_from_id("Qwen2.5-0.5B-Instruct") == 0.5
    assert V._params_b_from_id("mystery-model") == 0.0


def test_whisper_and_vision_footprints():
    assert V._whisper_mb("large-v3") == 3000
    assert V._whisper_mb("small") == 1000


def test_plan_sums_the_pod_stack(monkeypatch):
    _set_local(monkeypatch, "qwen2.5-14b-instruct")
    monkeypatch.setattr(cfg.live, "gpu_stt", True, raising=False)
    monkeypatch.setattr(cfg.stt, "model", "large-v3", raising=False)
    monkeypatch.setattr(cfg.stt, "partial_model", "small", raising=False)
    monkeypatch.setattr(cfg.stt, "partial_provider", "faster_whisper", raising=False)
    monkeypatch.setattr(cfg.vision, "mode", "local", raising=False)
    monkeypatch.setattr(cfg.vision, "use_gpu", True, raising=False)
    monkeypatch.setattr(cfg.vision, "provider", "qwen2_5_vl", raising=False)
    monkeypatch.setattr(cfg.vision, "qwen_vl_load_8bit", True, raising=False)
    plan = dict(V.estimate_plan())
    names = " ".join(plan.keys())
    assert "local-llm" in names and "stt-final" in names
    assert "stt-partial" in names and "vision" in names and "speculative-draft" in names
    total = sum(plan.values())
    # 14B(~9.6k) + large-v3(3k) + small(1k) + qwen-vl 8bit(3.5k) + draft(0.6k)
    assert 15000 < total < 22000, total


def test_over_budget_flagged(monkeypatch):
    monkeypatch.setattr(V, "_free_vram_mb", lambda: 8000)  # small GPU
    _set_local(monkeypatch, "qwen2.5-14b-instruct")
    monkeypatch.setattr(cfg.live, "gpu_stt", True, raising=False)
    monkeypatch.setattr(cfg.stt, "model", "large-v3", raising=False)
    r = V.preflight()
    assert r["ok"] is False and "exceeds" in r["warning"]


def test_no_gpu_is_ok(monkeypatch):
    monkeypatch.setattr(V, "_free_vram_mb", lambda: None)
    r = V.preflight()
    assert r["ok"] is True
