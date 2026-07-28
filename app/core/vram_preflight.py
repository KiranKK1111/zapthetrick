"""
VRAM preflight (Gap 4).

The pod now loads several models on ONE GPU — the local LLM (+ speculative draft
+ optional small tier), the STT final + a lighter partial, and the local vision
model. This module estimates their combined footprint from config and compares
it to the GPU's actual free VRAM at startup, logging a clear budget breakdown
and a WARNING (with what to trim) when the plan is over budget. Optionally it
GUARDS — disabling the least-critical extras (small tier → partial model →
draft) so a fresh deploy degrades instead of OOM-crash-looping.

Estimates are deliberately rough (a heuristic, ~±20%) — enough to catch "this
won't fit" before the models try to load, never a precise allocator. Fail-open:
any error → no preflight, today's behaviour.
"""
from __future__ import annotations

import logging
import re

from app.core.config_loader import cfg

log = logging.getLogger("zapthetrick.vram")

# Safety headroom kept free for activations / KV growth / fragmentation.
_HEADROOM_MB = 2000


def _params_b_from_id(model_id: str) -> float:
    """Best-effort parameter count (in billions) parsed from a model id/path
    ('qwen2.5-14b-instruct' → 14, 'Qwen2.5-0.5B' → 0.5). 0 when unknown."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id or "")
    return float(m.group(1)) if m else 0.0


def _llm_mb(model_id: str, *, quant: str = "q4") -> int:
    """Weights + a KV/activation allowance for a served GGUF. Q4 ≈ 0.6 GB/B."""
    pb = _params_b_from_id(model_id)
    if pb <= 0:
        return 0
    per_b = 600 if quant == "q4" else (2000 if quant == "fp16" else 1000)
    weights = pb * per_b
    return int(weights + 1200)  # ~1.2GB KV/activation allowance


def _whisper_mb(name: str) -> int:
    """faster-whisper footprint by model size (fp16 on GPU)."""
    n = (name or "").lower()
    if "large" in n or "distil-large" in n:
        return 3000
    if "medium" in n:
        return 1800
    if "small" in n:
        return 1000
    if "base" in n:
        return 600
    if "tiny" in n:
        return 300
    return 1500


def _vision_mb() -> int:
    v = cfg.vision
    if getattr(v, "mode", "local") != "local" or not getattr(v, "use_gpu", True):
        return 0
    prov = (getattr(v, "provider", "") or "").lower()
    if "qwen" in prov:
        return 3500 if getattr(v, "qwen_vl_load_8bit", False) else 7000
    if "minicpm" in prov:
        return 6000
    if "500m" in prov or "smolvlm_500m" in prov:
        return 1200
    if "smolvlm" in prov:
        return 4500
    return 1500


def estimate_plan() -> list[tuple[str, int]]:
    """The (component, MB) footprints implied by the current config. Deterministic."""
    plan: list[tuple[str, int]] = []
    try:
        loc = getattr(cfg.llm, "local", None)
        if loc is not None and getattr(loc, "enabled", False):
            plan.append((f"local-llm ({loc.model_id})", _llm_mb(loc.model_id)))
            small = (getattr(loc, "small_model_id", "") or "").strip()
            if small:
                plan.append((f"local-llm-small ({small})", _llm_mb(small)))
    except Exception:  # noqa: BLE001
        pass
    try:
        if bool(getattr(cfg.live, "gpu_stt", False)):
            plan.append((f"stt-final ({cfg.stt.model})", _whisper_mb(cfg.stt.model)))
            pm = (getattr(cfg.stt, "partial_model", "") or "").strip()
            if pm and pm != cfg.stt.model and \
                    (getattr(cfg.stt, "partial_provider", "") == "faster_whisper"):
                plan.append((f"stt-partial ({pm})", _whisper_mb(pm)))
    except Exception:  # noqa: BLE001
        pass
    try:
        vmb = _vision_mb()
        if vmb:
            plan.append((f"vision ({cfg.vision.provider})", vmb))
    except Exception:  # noqa: BLE001
        pass
    # Speculative draft (best-effort constant — GGUF served alongside).
    try:
        if bool(getattr(cfg.llm.local, "enabled", False)):
            plan.append(("speculative-draft", 600))
    except Exception:  # noqa: BLE001
        pass
    return plan


def _free_vram_mb() -> int | None:
    """Total GPU VRAM in MB (the budget target), or None when no CUDA GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(total // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None


def preflight() -> dict:
    """Compare the estimated plan to VRAM. Returns a report dict; never raises."""
    try:
        plan = estimate_plan()
        want = sum(mb for _, mb in plan)
        total = _free_vram_mb()
        report = {"plan": plan, "estimated_mb": want, "total_vram_mb": total,
                  "headroom_mb": _HEADROOM_MB, "ok": True, "warning": ""}
        if total is None:
            return report  # no GPU (CPU/dev) → nothing to check
        budget = total - _HEADROOM_MB
        report["ok"] = want <= budget
        lines = "; ".join(f"{n} ~{mb}MB" for n, mb in plan)
        if report["ok"]:
            log.info("VRAM preflight OK: plan ~%dMB + %dMB headroom <= %dMB "
                     "(%s)", want, _HEADROOM_MB, total, lines)
        else:
            report["warning"] = (
                f"VRAM plan ~{want}MB (+{_HEADROOM_MB}MB headroom) exceeds "
                f"{total}MB. Trim: drop llm.local.small_model_id, then "
                f"stt.partial_model, then a smaller vision model or 8-bit VLM.")
            log.warning("VRAM preflight OVER BUDGET: %s | %s",
                        report["warning"], lines)
        return report
    except Exception:  # noqa: BLE001
        return {"ok": True, "warning": "", "plan": [], "estimated_mb": 0,
                "total_vram_mb": None, "headroom_mb": _HEADROOM_MB}


__all__ = ["preflight", "estimate_plan"]
