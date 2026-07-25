"""Canonical model identity (vNext §2.4).

Every catalog row is a `(platform, model_id)` pair, so the SAME weights served by
Groq / Cerebras / NVIDIA-NIM / OpenRouter / HuggingFace look like different
candidates — and a provider failure can jump the ladder to a genuinely different
model when the *same* one was available next door. This module normalizes a
`(platform, model_id)` to a provider-independent **canonical identity**:

    CanonicalId(family, size_b, version, quant)

so `llama-3.3-70b-versatile`@groq and `Llama-3.3-70B-Instruct`@cerebras are ONE
model, while a 4-bit AWQ serving is (correctly) NOT the same identity as the fp8
serving — quantization is part of identity, and its scorecard proves whatever
difference exists.

This is the foundation for two-stage selection + same-model failover (§2.4),
quota spread across a model's providers (§2.7), and gauntlet scorecards keyed by
identity (§2.5). Pure + fail-open: an unrecognizable id yields a STABLE `unknown`
identity keyed by the raw id (nothing is ever lost, nothing crashes). Reuses the
size/MoE heuristics already in `catalog.py` (intra-`llm` — no new edge). The
name/quant/suffix tables are DATA lookups, not intent (they stay literal).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── DATA tables (lookups, updatable without touching logic) ──────────────────
# Quantization tokens → a normalized quant tag. Order matters (longest first).
_QUANT_TOKENS: tuple[tuple[str, str], ...] = (
    ("int4", "int4"), ("int8", "int8"), ("fp8", "fp8"), ("fp16", "fp16"),
    ("bf16", "bf16"), ("awq", "awq"), ("gptq", "gptq"), ("gguf", "gguf"),
    ("4bit", "int4"), ("8bit", "int8"), ("q4", "int4"), ("q8", "int8"),
    ("w4a16", "int4"), ("w8a8", "int8"),
)
# Serving/marketing suffix WORDS that describe a deployment, not the weights —
# stripped from the family stem so variants of one model collapse together.
_SERVING_SUFFIXES: frozenset[str] = frozenset({
    "instruct", "chat", "it", "base", "versatile", "turbo", "preview", "hf",
    "free", "latest", "online", "fast", "beta", "alpha", "stable", "release",
    "text", "generate", "completion", "specdec", "tool", "tools",
})
# A version-like token: v3, 3.3, 2.5, 4.1 — matched AFTER the size token is
# removed, so the remaining number is the version, not the parameter count.
_VERSION_RE = re.compile(r"(?<![a-z0-9])v?(\d+(?:\.\d+){0,2})(?![a-z])", re.I)
_SEP_RE = re.compile(r"[\s_/.:@+-]+")


@dataclass(frozen=True)
class CanonicalId:
    family: str            # "llama" / "qwen-coder" / "unknown:<raw>"
    size_b: float | None   # approx total params in billions (None = undisclosed)
    version: str           # "3.3" / "2.5" / "" (none)
    quant: str             # "awq" / "fp8" / "int4" / "" (default/full precision)

    def key(self) -> str:
        sz = f"{self.size_b:g}" if self.size_b is not None else "?"
        return f"{self.family}|{sz}|{self.version or '-'}|{self.quant or '-'}"

    def __str__(self) -> str:
        return self.key()


def _extract_quant(s: str) -> str:
    for tok, tag in _QUANT_TOKENS:
        if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", s):
            return tag
    return ""


def _size_label(model_id: str) -> str | None:
    from app.llm import catalog
    return catalog.detect_param_count(model_id)


def _size_billions(model_id: str) -> float | None:
    from app.llm import catalog
    return catalog._size_billions(model_id)


def _extract_version(stem_no_size: str) -> str:
    m = _VERSION_RE.search(stem_no_size)
    return m.group(1) if m else ""


def _family_stem(s: str, size_label: str | None, quant: str,
                 version: str) -> str:
    """The family name: the id with the org prefix, size token, quant tokens,
    version, and serving-suffix words removed; separators collapsed to '-'. A
    genuine variant marker (coder/vl/vision/math/...) is NOT a serving suffix, so
    it stays part of the family."""
    body = s
    if size_label:
        body = re.sub(rf"(?<![a-z0-9]){re.escape(size_label.lower())}(?![a-z0-9])",
                      " ", body)
    # active-param token (a22b) is a serving detail of an MoE, drop it too.
    body = re.sub(r"(?<![a-z0-9])a\d+(?:\.\d+)?b(?![a-z0-9])", " ", body)
    for tok, _ in _QUANT_TOKENS:
        body = re.sub(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", " ", body)
    words = [w for w in _SEP_RE.split(body) if w]
    kept = [w for w in words if w not in _SERVING_SUFFIXES]
    # Drop the bare version number token from the family (it's its own field),
    # but keep tokens like "3.3" only when they equal the extracted version.
    if version:
        kept = [w for w in kept if w.strip("v") != version]
    stem = "-".join(kept).strip("-")
    return stem


def canonicalize(platform: str, model_id: str,
                 meta: dict | None = None) -> CanonicalId:
    """Normalize a `(platform, model_id)` to its provider-independent identity.
    Never raises → a stable `unknown:<raw>` family for anything unrecognizable."""
    raw = (model_id or "").strip()
    try:
        s = raw.lower()
        if "/" in s:                       # strip an org/namespace prefix
            s = s.rsplit("/", 1)[-1]
        quant = _extract_quant(s)
        size_label = _size_label(raw)
        size_b = _size_billions(raw)
        # Remove the size token before reading the version so "70b" isn't misread.
        no_size = s
        if size_label:
            no_size = re.sub(
                rf"(?<![a-z0-9]){re.escape(size_label.lower())}(?![a-z0-9])",
                " ", no_size)
        version = _extract_version(no_size)
        family = _family_stem(s, size_label, quant, version)
        if not family:
            family = f"unknown:{raw.lower()}"
        return CanonicalId(family=family, size_b=size_b, version=version,
                           quant=quant)
    except Exception:  # noqa: BLE001 — identity is best-effort, never fatal
        return CanonicalId(family=f"unknown:{raw.lower()}", size_b=None,
                           version="", quant="")


def identity_key(platform: str, model_id: str,
                 meta: dict | None = None) -> str:
    """The canonical key when the flag is ON, else the raw `(platform, model_id)`
    so callers get today's per-(provider,model) behaviour with `canonical_
    identity` off."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.routing, "canonical_identity", False)):
            return f"{platform}:{model_id}"
    except Exception:  # noqa: BLE001
        return f"{platform}:{model_id}"
    return canonicalize(platform, model_id, meta).key()


def build_provider_index(
    rows: "list[tuple[str, str, object]]",
) -> dict[str, list[tuple[str, str, object]]]:
    """Group catalog/candidate rows ``(platform, model_id, ref)`` by canonical
    identity key → the providers serving each identity. `ref` is opaque (a
    model_db_id, a candidate dict, …). Pure; the caller supplies the rows so this
    module imports no DB."""
    index: dict[str, list[tuple[str, str, object]]] = {}
    for platform, model_id, ref in rows:
        cid = canonicalize(platform, model_id)
        index.setdefault(cid.key(), []).append((platform, model_id, ref))
    return index


def providers_for(cid: CanonicalId,
                  index: dict[str, list]) -> list[tuple[str, str, object]]:
    """The rows serving `cid` in a `build_provider_index` result (or [])."""
    return list(index.get(cid.key(), []))


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, "canonical_identity", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "CanonicalId", "canonicalize", "identity_key", "build_provider_index",
    "providers_for", "enabled",
]
