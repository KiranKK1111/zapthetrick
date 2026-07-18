"""Local inference optimization — quantization / kv-cache / batching planner
(roadmap Phase 5 #15).

This deployment answers via remote providers and runs only local STT + embeddings
(no in-repo local LLM generation runtime). So a *genuine* minimal version of
"local inference optimization" is the PLANNER a local runtime (llama.cpp / vLLM /
TGI) would consume: given the GPU's VRAM and a model's parameter count, pick the
heaviest quantization that fits with room for the KV cache, and a batch size that
uses the remaining headroom — the real levers of local-inference throughput.

It computes settings; it does not execute a model (there is nothing local to
execute). Pure arithmetic, deterministic, fail-open. If/when a local generation
backend is added, it consumes `LocalInferenceConfig` directly.
"""
from __future__ import annotations

from dataclasses import dataclass

# Quantization → bytes-per-weight.
_QUANT_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
# Preference order: fp16 (best quality) first, fall back to smaller as needed.
_QUANT_ORDER = ["fp16", "int8", "int4"]

# Fraction of VRAM we refuse to allocate (driver, fragmentation, activations).
_SAFETY = 0.10
# Rough KV-cache bytes per token per billion params at fp16 (order-of-magnitude).
_KV_BYTES_PER_TOK_PER_B = 2.0 * 1024  # ~2KB/token/B — conservative


@dataclass
class LocalInferenceConfig:
    quantization: str          # fp16 | int8 | int4 | "none" (won't fit)
    weights_gb: float          # VRAM the weights occupy at this quant
    kv_cache_mb: float         # VRAM budgeted for the KV cache
    max_batch: int             # recommended concurrent sequences
    context_tokens: int        # context length the KV budget assumes
    fits: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "quantization": self.quantization,
            "weights_gb": round(self.weights_gb, 2),
            "kv_cache_mb": round(self.kv_cache_mb, 1),
            "max_batch": self.max_batch,
            "context_tokens": self.context_tokens,
            "fits": self.fits,
            "reason": self.reason,
        }


def _weights_gb(params_b: float, quant: str) -> float:
    return params_b * 1e9 * _QUANT_BYTES[quant] / (1024 ** 3)


def recommend(vram_gb: float, params_b: float, *,
              context_tokens: int = 4096,
              prefer: str | None = None) -> LocalInferenceConfig:
    """Pick a quantization + KV/batch plan for `params_b`-billion-param model on
    a GPU with `vram_gb` of memory. Chooses the highest-quality quant that leaves
    room for at least one full-context KV cache; sizes the batch from what's
    left. Fail-open: bad inputs → a not-fitting config, never raises."""
    try:
        usable = max(0.0, float(vram_gb) * (1.0 - _SAFETY))
        order = ([prefer] + [q for q in _QUANT_ORDER if q != prefer]) \
            if prefer in _QUANT_BYTES else _QUANT_ORDER

        kv_one_seq_gb = (_KV_BYTES_PER_TOK_PER_B * params_b * context_tokens) \
            / (1024 ** 3)

        for quant in order:
            w_gb = _weights_gb(params_b, quant)
            if w_gb + kv_one_seq_gb > usable:
                continue                       # can't even fit weights + 1 KV
            headroom_gb = usable - w_gb
            max_batch = max(1, int(headroom_gb // max(kv_one_seq_gb, 1e-6)))
            kv_budget_mb = max_batch * kv_one_seq_gb * 1024
            return LocalInferenceConfig(
                quantization=quant, weights_gb=w_gb, kv_cache_mb=kv_budget_mb,
                max_batch=max_batch, context_tokens=context_tokens, fits=True,
                reason=f"{quant} weights {w_gb:.1f}GB + {max_batch}x KV in "
                       f"{usable:.1f}GB usable")
        return LocalInferenceConfig(
            "none", _weights_gb(params_b, "int4"), 0.0, 0, context_tokens, False,
            f"{params_b}B won't fit in {vram_gb}GB even at int4")
    except Exception as exc:  # noqa: BLE001
        return LocalInferenceConfig("none", 0.0, 0.0, 0, context_tokens, False,
                                    f"planner error: {str(exc)[:80]}")


def detect_vram_gb() -> float | None:
    """Best-effort VRAM read via the capability probe / torch. None when no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


def recommend_for_local_gpu(params_b: float, *, context_tokens: int = 4096
                            ) -> LocalInferenceConfig | None:
    """Plan for the actual local GPU, or None when there is no CUDA device."""
    vram = detect_vram_gb()
    if vram is None:
        return None
    return recommend(vram, params_b, context_tokens=context_tokens)


__all__ = [
    "LocalInferenceConfig", "recommend", "recommend_for_local_gpu",
    "detect_vram_gb",
]
