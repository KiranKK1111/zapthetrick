"""Per-(model-identity, profile) measured scorecards (vNext §2.6).

This router's unfair advantage: the sandbox reports, per model, whether generated
code *actually compiled and passed examples* (Stage-4 verify lanes) and whether a
structured call validated. We keep an EWMA of **verify-pass rate**, **repair-
trigger rate**, and **schema-retry rate** per `(CanonicalId key, task profile)`,
fed continuously from the §8.9 ledger + the verify lanes — so `coder`/`json`
rankings stop being an opinion and become *measured on this deployment's traffic*.

`verify_pass_rate(...)` is the additive score input the router consumes (weight 0
by default → today's ranking, byte-identical). Process-wide, thread-safe, bounded;
survives across requests, not restarts (a speed/quality signal, not a source of
truth). Seeded optimistic (1.0) so an UNMEASURED model is never penalized — only
demonstrated failures lower it.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

_ALPHA = 0.2        # EWMA weight of the newest observation
_MAX = 4096         # bounded (identity, profile) cells
_lock = threading.RLock()


@dataclass
class _Card:
    verify_pass: float = 1.0     # seeded optimistic (unmeasured ≠ penalized)
    repair_rate: float = 0.0
    schema_retry: float = 0.0
    n: int = 0


_store: "OrderedDict[tuple[str, str], _Card]" = OrderedDict()


def _cell(identity_key: str, profile: str) -> _Card:
    k = (identity_key, profile)
    card = _store.get(k)
    if card is None:
        card = _Card()
        _store[k] = card
        while len(_store) > _MAX:
            _store.popitem(last=False)
    _store.move_to_end(k)
    return card


def _ewma(prev: float, obs: float) -> float:
    return (1 - _ALPHA) * prev + _ALPHA * obs


def record_verify_outcome(identity_key: str, profile: str | None, *,
                          passed: bool, repaired: bool = False,
                          schema_retried: bool = False) -> None:
    """Fold one sandbox/verify outcome into the (identity, profile) scorecard.
    No-op when there's no profile. Never raises."""
    if not identity_key or not profile:
        return
    try:
        with _lock:
            c = _cell(identity_key, profile)
            c.verify_pass = _ewma(c.verify_pass, 1.0 if passed else 0.0)
            c.repair_rate = _ewma(c.repair_rate, 1.0 if repaired else 0.0)
            c.schema_retry = _ewma(c.schema_retry, 1.0 if schema_retried else 0.0)
            c.n += 1
    except Exception:  # noqa: BLE001 — telemetry never breaks a turn
        pass


def verify_pass_rate(identity_key: str, profile: str | None) -> float:
    """The measured verify-pass EWMA for this (identity, profile) — 1.0 when
    unmeasured (optimistic, so a new model isn't penalized)."""
    if not identity_key or not profile:
        return 1.0
    with _lock:
        c = _store.get((identity_key, profile))
        return c.verify_pass if c is not None else 1.0


def card(identity_key: str, profile: str | None) -> dict:
    with _lock:
        c = _store.get((identity_key, profile or ""))
        if c is None:
            return {"verify_pass": 1.0, "repair_rate": 0.0,
                    "schema_retry": 0.0, "n": 0}
        return {"verify_pass": round(c.verify_pass, 4),
                "repair_rate": round(c.repair_rate, 4),
                "schema_retry": round(c.schema_retry, 4), "n": c.n}


def clear() -> None:
    with _lock:
        _store.clear()


def profile_verify_weight() -> float:
    """Additive weight of the verify-pass penalty term (0 → off, today's score)."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.routing, "task_profiles", False)):
            return 0.0
        return float(getattr(cfg.routing, "profile_verify_weight", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


__all__ = ["record_verify_outcome", "verify_pass_rate", "card", "clear",
           "profile_verify_weight"]
