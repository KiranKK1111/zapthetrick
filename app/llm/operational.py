"""Operational modes — unified offline-first + reproducible layer (P5 #27).

Two operator-facing modes were previously scattered / partial:

  * OFFLINE-FIRST — prefer local models and avoid egress to cloud providers
    (privacy / air-gapped / cost). Surfaced here as a routing preference the
    router can consult; fail-open so an offline-only deployment with no local
    model still answers rather than dead-ends.
  * REPRODUCIBLE  — deterministic sampling: temperature pinned to 0 and a fixed
    integer `seed` sent to providers that honour it, so the same prompt yields
    the same answer (evals / debugging / audits).

This module is the single source of truth for both. `OperationalMode.current()`
reads config; `apply_to_options(options)` stamps the sampling params;
`prefer_local`/`allow_cloud` are the routing hooks. Both modes default OFF →
byte-identical to today. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

# Providers considered LOCAL (no egress). Everything else is cloud.
_LOCAL_PLATFORMS = {"ollama", "local", "llamacpp", "llama_cpp", "vllm"}

_DEFAULT_SEED = 1234


def offline_first() -> bool:
    """`cfg.llm.offline_first` — default OFF (behaviour-changing, opt-in)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "llm", None), "offline_first", False))
    except Exception:  # noqa: BLE001
        return False


def reproducible() -> bool:
    """`cfg.llm.reproducible` — default OFF (opt-in determinism)."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "llm", None), "reproducible", False))
    except Exception:  # noqa: BLE001
        return False


def reproducible_seed() -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(getattr(cfg, "llm", None), "reproducible_seed",
                           _DEFAULT_SEED) or _DEFAULT_SEED)
    except Exception:  # noqa: BLE001
        return _DEFAULT_SEED


def is_local_platform(platform: str | None) -> bool:
    return (platform or "").strip().lower() in _LOCAL_PLATFORMS


@dataclass
class OperationalMode:
    offline: bool = False
    reproducible: bool = False
    seed: int = _DEFAULT_SEED

    @classmethod
    def current(cls) -> "OperationalMode":
        return cls(offline=offline_first(), reproducible=reproducible(),
                   seed=reproducible_seed())

    def prefer_local(self) -> bool:
        """Router hook: True when local models should sort ahead of cloud."""
        return self.offline

    def allow_cloud(self) -> bool:
        """Router hook: cloud is still allowed as a LAST RESORT even offline
        (fail-open — an offline deployment with no local model must still work)."""
        return True

    def apply_to_options(self, options: dict) -> dict:
        """Stamp reproducible sampling params onto a provider `options` dict.
        No-op when reproducible mode is off. Never overrides an explicit caller
        temperature of 0 or an explicit seed."""
        try:
            if not self.reproducible:
                return options
            if options.get("temperature") is None:
                options["temperature"] = 0
            if options.get("seed") is None:
                options["seed"] = self.seed
        except Exception:  # noqa: BLE001
            pass
        return options

    def snapshot(self) -> dict:
        return {"offline": self.offline, "reproducible": self.reproducible,
                "seed": self.seed}


def apply_to_options(options: dict) -> dict:
    """Convenience: apply the CURRENT operational mode to `options`."""
    return OperationalMode.current().apply_to_options(options)


def order_by_locality(platforms: list[str]) -> list[str]:
    """Stable sort putting local platforms first (used under offline-first)."""
    try:
        return sorted(platforms, key=lambda p: (0 if is_local_platform(p) else 1))
    except Exception:  # noqa: BLE001
        return platforms


__all__ = [
    "OperationalMode", "offline_first", "reproducible", "reproducible_seed",
    "is_local_platform", "apply_to_options", "order_by_locality",
]
