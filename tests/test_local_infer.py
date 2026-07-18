"""Local inference optimization planner (roadmap Phase 5 #15).

Pins the quantization/KV/batch planner arithmetic: bigger VRAM → higher-quality
quant + more batch; a too-big model doesn't fit; degrade fp16 → int8 → int4.
"""
from __future__ import annotations

from app.llm.local_infer import LocalInferenceConfig, recommend


def test_small_model_big_gpu_uses_fp16_and_batches():
    cfg = recommend(vram_gb=24, params_b=7, context_tokens=4096)
    assert isinstance(cfg, LocalInferenceConfig)
    assert cfg.fits and cfg.quantization == "fp16"
    assert cfg.max_batch >= 1
    assert cfg.weights_gb < 24


def test_tight_vram_degrades_quantization():
    # A 13B model won't fit fp16 (26GB) in 16GB, but int8 (13GB) can.
    cfg = recommend(vram_gb=16, params_b=13, context_tokens=2048)
    assert cfg.fits and cfg.quantization in ("int8", "int4")


def test_too_big_model_does_not_fit():
    cfg = recommend(vram_gb=8, params_b=180, context_tokens=4096)
    assert not cfg.fits and cfg.quantization == "none"


def test_more_vram_gives_more_batch():
    small = recommend(vram_gb=16, params_b=7).max_batch
    big = recommend(vram_gb=48, params_b=7).max_batch
    assert big > small


def test_prefer_quantization_respected():
    cfg = recommend(vram_gb=24, params_b=7, prefer="int4")
    assert cfg.quantization == "int4" and cfg.fits


def test_failopen_on_bad_input():
    cfg = recommend(vram_gb="oops", params_b=7)   # type: ignore[arg-type]
    assert isinstance(cfg, LocalInferenceConfig) and not cfg.fits


def test_as_dict_shape():
    d = recommend(vram_gb=24, params_b=7).as_dict()
    assert set(d) >= {"quantization", "weights_gb", "kv_cache_mb", "max_batch",
                      "context_tokens", "fits"}
