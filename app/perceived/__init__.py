"""Perceived-speed subsystem (perceived-speed spec).

Reduces actual + perceived latency: a shared pooled HTTP client (in
`app.core.http_pool`), a speculation budget + kill-switch (`budget`), intent
prediction + prefetch (`prefetch`), and a predictive answer cache (`cache`).

Everything is flag-gated by `cfg.perceived` and fails open — with
`speculation_enabled=False` (the default) none of the speculative paths run and
behavior is byte-for-byte today's.
"""
from .budget import SpeculationBudget, budget, speculation_enabled

__all__ = ["SpeculationBudget", "budget", "speculation_enabled"]
