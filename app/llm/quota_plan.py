"""Proactive free-quota planning (vNext §2.7) — the planned half of quota mgmt.

`quota_manager.py` reacts to a draining free tier per PROVIDER. This adds the
§2.7 planning layer the router consumes when `routing.quota_planning` is on:

  * **Per-`(provider, key_id)` daily ledgers** — the spread balances across a
    provider's multiple keys, not just providers. Seeded from `DEFAULTS` (data),
    injectable clock, window-rolling on the provider's real boundary.
  * **Header-correction** — reconcile a ledger from a provider's own
    `Retry-After` / `x-ratelimit-remaining` / `x-ratelimit-reset` headers, so the
    seeded estimate is replaced by ground truth when the provider reports it.
  * **Reserve** (`routing.live_reserve`) — hold a slice (default 30%) of each
    key's daily quota for Live; a non-Live turn sees headroom MINUS the reserve,
    a Live turn sees it all. The reserve is released in the final hours before
    reset so it never goes to waste.
  * **Spread** (`routing.spread`) — rank the keys/providers of one canonical
    model best-first by remaining headroom, so routine traffic rotates and all
    quotas stay alive through the day instead of exhausting the favourite.

Everything is fail-open: any error yields "no signal" so routing is unchanged.
Persistence mirrors `ratelimit.py` (best-effort Postgres, rehydrate on boot) and
reuses its usage table with a distinct `kind`, so no new migration is needed.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.llm.quota_manager import DAY, DEFAULTS

log = logging.getLogger(__name__)

# Default fraction of a key's daily quota reserved for Live (released near reset).
_RESERVE_FRACTION = 0.30
# Release the reserve when within this long of the window boundary (nothing
# scheduled → don't waste it): the last 3 hours of the day.
_RESERVE_RELEASE_S = 3 * 3600.0


def _cfg_bool(name: str, default: bool = False) -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, name, default))
    except Exception:  # noqa: BLE001
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        from app.core.config_loader import cfg
        return float(getattr(cfg.routing, name, default))
    except Exception:  # noqa: BLE001
        return default


def enabled() -> bool:
    return _cfg_bool("quota_planning", False)


def spread_enabled() -> bool:
    return enabled() and _cfg_bool("spread", False)


def reserve_enabled() -> bool:
    return enabled() and _cfg_bool("live_reserve", False)


def reserve_fraction() -> float:
    f = _cfg_float("live_reserve_fraction", _RESERVE_FRACTION)
    return max(0.0, min(0.9, f))


@dataclass
class _Ledger:
    limit: int              # requests per window (0 = unlimited / unknown)
    window_s: float
    used: int = 0
    window_start: float = 0.0
    # When set (epoch s), a header/429 told us we're blocked until then.
    blocked_until: float = 0.0
    # §2.7 Live session-plan reservation: expected spend held for a pinned Live
    # session (Component F). Subtracted from headroom so other traffic can't eat
    # the session's budget mid-interview; released when the session ends.
    reserved: int = 0


class QuotaPlanner:
    """Per-`(provider, key_id)` daily ledgers with reserve + spread + header
    reconciliation. Deterministic, injectable clock, in-process, fail-open."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._l: dict[str, _Ledger] = {}
        self._now = now or time.time

    # ── keys / seeding ────────────────────────────────────────────────────
    @staticmethod
    def _key(provider: str, key_id: int | None) -> str:
        return f"{(provider or '').strip().lower()}:{key_id if key_id is not None else '_'}"

    def _seed(self, provider: str, key_id: int | None) -> _Ledger:
        prov = (provider or "").strip().lower()
        limit, window = DEFAULTS.get(prov, (0, DAY))
        led = _Ledger(limit=int(limit), window_s=float(window),
                      window_start=self._now())
        self._l[self._key(prov, key_id)] = led
        return led

    def _get(self, provider: str, key_id: int | None) -> _Ledger:
        led = self._l.get(self._key(provider, key_id))
        if led is None:
            led = self._seed(provider, key_id)
        self._roll(led)
        return led

    def _roll(self, led: _Ledger) -> None:
        now = self._now()
        if led.window_s > 0 and now - led.window_start >= led.window_s:
            elapsed = now - led.window_start
            led.window_start += int(elapsed // led.window_s) * led.window_s
            led.used = 0
            led.blocked_until = 0.0

    # ── record + query ────────────────────────────────────────────────────
    def record(self, provider: str, key_id: int | None = None, n: int = 1) -> None:
        led = self._get(provider, key_id)
        led.used += n

    def headroom(self, provider: str, key_id: int | None = None) -> int | None:
        """Remaining requests this window, NET of any Live reservation; None when
        unlimited/unknown."""
        led = self._get(provider, key_id)
        if led.limit <= 0:
            return None
        if led.blocked_until and self._now() < led.blocked_until:
            return 0
        return max(0, led.limit - led.used - led.reserved)

    def reserve(self, provider: str, key_id: int | None, n: int) -> None:
        """Hold `n` requests of a pinned Live session's expected spend (§2.7 F).
        Bounded so a reservation can never exceed the remaining window."""
        if n <= 0:
            return
        led = self._get(provider, key_id)
        room = max(0, led.limit - led.used - led.reserved) if led.limit > 0 else n
        led.reserved += min(n, room) if led.limit > 0 else n

    def release(self, provider: str, key_id: int | None, n: int) -> None:
        """Release a reservation (session ended). Never goes negative."""
        if n <= 0:
            return
        led = self._get(provider, key_id)
        led.reserved = max(0, led.reserved - n)

    def headroom_fraction(self, provider: str, key_id: int | None = None,
                          *, for_live: bool = False) -> float:
        """0..1 remaining fraction, with the Live RESERVE applied for non-Live
        turns (unless we're near reset, when the reserve is released). Unknown /
        unlimited → 1.0 (no signal)."""
        led = self._get(provider, key_id)
        if led.limit <= 0:
            return 1.0
        h = self.headroom(provider, key_id) or 0
        frac = h / led.limit
        if not for_live and reserve_enabled():
            # Withhold the reserve slice unless we're in the release window.
            resets_in = (led.window_start + led.window_s) - self._now()
            if resets_in > _RESERVE_RELEASE_S:
                frac = max(0.0, frac - reserve_fraction())
        return max(0.0, min(1.0, frac))

    def exhausted(self, provider: str, key_id: int | None = None,
                  *, for_live: bool = False) -> bool:
        h = self.headroom(provider, key_id)
        if h is None:
            return False
        if for_live:
            return h <= 0
        # A non-Live turn is "exhausted" once it has eaten everything but the
        # reserve (so the reserve is genuinely held back), unless near reset.
        led = self._get(provider, key_id)
        resets_in = (led.window_start + led.window_s) - self._now()
        floor = 0
        if reserve_enabled() and resets_in > _RESERVE_RELEASE_S:
            floor = int(reserve_fraction() * led.limit)
        return h <= floor

    def next_reset(self, provider: str, key_id: int | None = None) -> float | None:
        led = self._get(provider, key_id)
        if led.window_s <= 0:
            return None
        return led.window_start + led.window_s

    # ── header-correction ─────────────────────────────────────────────────
    def reconcile_headers(self, provider: str, key_id: int | None,
                          headers: dict) -> None:
        """Update a ledger from a provider's own rate-limit headers (case-
        insensitive). Recognizes `retry-after` (block until now+seconds),
        `x-ratelimit-remaining[-requests]` (authoritative remaining → used), and
        `x-ratelimit-reset[-requests]` (seconds-until or epoch reset). Missing /
        unparseable headers leave the seeded estimate (fail-open)."""
        try:
            h = {str(k).lower(): v for k, v in (headers or {}).items()}
        except Exception:  # noqa: BLE001
            return
        if not h:
            return
        led = self._get(provider, key_id)
        now = self._now()

        ra = _num(h.get("retry-after"))
        if ra is not None and ra > 0:
            led.blocked_until = now + ra

        rem = _num(h.get("x-ratelimit-remaining-requests")
                   or h.get("x-ratelimit-remaining"))
        if rem is not None and led.limit > 0:
            led.used = max(0, led.limit - int(rem))

        rst = _num(h.get("x-ratelimit-reset-requests")
                   or h.get("x-ratelimit-reset"))
        if rst is not None and rst > 0:
            # Heuristic: a small value is "seconds from now"; a large one is an
            # absolute epoch. Either way, align the window so it rolls then.
            reset_at = now + rst if rst < led.window_s * 4 else rst
            led.window_start = reset_at - led.window_s

    # ── spread ────────────────────────────────────────────────────────────
    def rank_keys(self, provider: str, key_ids: list[int]) -> list[int]:
        """Order a provider's keys best-first by remaining headroom (spread): a
        drained key sinks so routine traffic rotates and all keys stay alive."""
        def score(kid: int) -> tuple[int, float]:
            h = self.headroom(provider, kid)
            if h is None:
                return (0, 0.0)            # unlimited/unknown first
            if h <= 0:
                return (2, 0.0)            # exhausted last
            return (1, -float(h))          # more headroom = earlier
        return sorted(key_ids, key=score)

    def spread_penalty(self, provider: str, key_id: int | None = None) -> float:
        """0..1 additive penalty for a draining key (1 - headroom fraction), so
        the router balances load across a model's providers/keys. 0 when spread
        is off or headroom is unknown/full."""
        if not spread_enabled():
            return 0.0
        led = self._get(provider, key_id)
        if led.limit <= 0:
            return 0.0
        h = self.headroom(provider, key_id) or 0
        return max(0.0, min(1.0, 1.0 - h / led.limit))

    def provider_signal(self, provider: str, *,
                        for_live: bool = False) -> tuple[float, bool] | None:
        """Aggregate a provider's per-key ledgers into the router's per-provider
        signal: ``(best-key reserve-adjusted headroom fraction, all-keys-
        exhausted)``. None when the provider has no known ledger (→ no signal, the
        router treats it as full headroom). The BEST key wins because the router
        will pick that key; reserve/exhaustion honour `for_live`."""
        prov = (provider or "").strip().lower()
        keys = [k for k in self._l if k.startswith(f"{prov}:")]
        if not keys:
            return None
        best = 0.0
        all_exhausted = True
        seen = False
        for k in keys:
            led = self._l[k]
            if led.limit <= 0:
                return None                # unlimited/unknown → no signal
            seen = True
            _, _, kid = k.partition(":")
            kid_i = int(kid) if kid.isdigit() else None
            frac = self.headroom_fraction(prov, kid_i, for_live=for_live)
            best = max(best, frac)
            if not self.exhausted(prov, kid_i, for_live=for_live):
                all_exhausted = False
        if not seen:
            return None
        return (best, all_exhausted)

    # ── introspection ─────────────────────────────────────────────────────
    def snapshot(self) -> list[dict]:
        out = []
        for k, led in self._l.items():
            self._roll(led)
            prov, _, kid = k.partition(":")
            out.append({
                "provider": prov, "key_id": kid,
                "limit": led.limit, "used": led.used,
                "headroom": (max(0, led.limit - led.used) if led.limit > 0 else None),
                "resets_at": (led.window_start + led.window_s
                              if led.window_s > 0 else None),
                "blocked_until": led.blocked_until or None,
            })
        return out

    def clear(self) -> None:
        self._l.clear()


def _num(v) -> float | None:
    """Parse a header value to a float (handles '30', '30s', '1.5'). None on any
    non-numeric input."""
    if v is None:
        return None
    try:
        s = str(v).strip().lower().rstrip("s")
        return float(s)
    except (TypeError, ValueError):
        return None


_planner = QuotaPlanner()


def quota_planner() -> QuotaPlanner:
    return _planner


# --------------------------------------------------------------------------- #
# Best-effort Postgres persistence (mirrors ratelimit.py) — reuses the generic
# `llm_settings` KV table with a single JSON blob, so NO new migration/table is
# needed. Durability only: the in-memory ledger is authoritative and works with
# no DB at all (fail-open), persistence just survives a restart/redeploy.
# --------------------------------------------------------------------------- #
_SETTING_KEY = "quota_plan_ledgers"


def _fire(coro) -> None:
    import asyncio
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()  # no loop (unit test) — in-memory is enough


def _serialize() -> str:
    import json
    rows = []
    for k, led in _planner._l.items():
        rows.append({"k": k, "l": led.limit, "w": led.window_s,
                     "u": led.used, "s": led.window_start,
                     "b": led.blocked_until})
    return json.dumps(rows, separators=(",", ":"))


async def _persist() -> None:
    if not enabled():
        return
    try:
        from sqlalchemy.dialects.postgresql import insert
        from storage.db import get_session_factory
        from storage.models import LLMSetting
        factory = get_session_factory()
        if factory is None:
            return
        blob = _serialize()
        async with factory() as session:
            stmt = insert(LLMSetting).values(key=_SETTING_KEY, value=blob) \
                .on_conflict_do_update(index_elements=["key"],
                                       set_={"value": blob})
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        log.debug("quota-plan persist failed: %s", exc)


async def rehydrate() -> int:
    """Restore ledgers from Postgres on boot (best-effort). Returns how many were
    loaded; 0 on any error / no DB / disabled. Call from app startup."""
    if not enabled():
        return 0
    try:
        import json

        from sqlalchemy import select
        from storage.db import get_session_factory
        from storage.models import LLMSetting
        factory = get_session_factory()
        if factory is None:
            return 0
        async with factory() as session:
            row = (await session.execute(
                select(LLMSetting).where(LLMSetting.key == _SETTING_KEY)
            )).scalar_one_or_none()
        if row is None or not row.value:
            return 0
        loaded = 0
        for r in json.loads(row.value):
            _planner._l[r["k"]] = _Ledger(
                limit=int(r["l"]), window_s=float(r["w"]), used=int(r["u"]),
                window_start=float(r["s"]), blocked_until=float(r.get("b", 0.0)))
            loaded += 1
        return loaded
    except Exception as exc:  # noqa: BLE001
        log.debug("quota-plan rehydrate failed: %s", exc)
        return 0


_last_persist = 0.0
_PERSIST_EVERY_S = 60.0


def record_success(provider: str, key_id: int | None) -> None:
    """Called from the engine after a successful completion (planning on only).
    Persists the ledger at most once per minute (best-effort, off the hot path)."""
    if not enabled():
        return
    try:
        _planner.record(provider, key_id, 1)
        global _last_persist
        mono = time.monotonic()
        if mono - _last_persist >= _PERSIST_EVERY_S:
            _last_persist = mono
            _fire(_persist())
    except Exception:  # noqa: BLE001
        pass


def reconcile(provider: str, key_id: int | None, headers: dict | None) -> None:
    """Engine hook — reconcile the ledger from the last response's headers."""
    if not enabled() or not headers:
        return
    try:
        _planner.reconcile_headers(provider, key_id, headers)
    except Exception:  # noqa: BLE001
        pass


def reset_for_tests() -> None:
    _planner.clear()


__all__ = ["QuotaPlanner", "quota_planner", "enabled", "spread_enabled",
           "reserve_enabled", "reserve_fraction", "record_success", "reconcile",
           "reset_for_tests"]
