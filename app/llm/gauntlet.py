"""Onboarding gauntlet (vNext §2.5) — discovery no longer implies eligibility.

A newly-discovered `(CanonicalId, provider)` pair is a *candidate*, not a
proven model. Until it passes a probe battery it is **quarantined**: not
selectable above the wide-fallback rung (T3), so an unknown can never outrank a
known-good model — but the never-empty ladder can still reach it in extremis.

The probe battery (run on the idle/background lane §9.1, its spend drawn from
the §2.7 spread budget so it never eats Live's reserve) measures the stats the
§2.6 scorer consumes: schema reliability, an instruction-quality prior, a
sandbox-compiled `coder` prior, effective context, TTFT/TPS, and adapter
capabilities. **The battery's results ARE the initial scorecard.** A rolling
re-probe (monthly) and an error-signature change trigger a fresh run.

The probes themselves are INJECTED (`ProbeSuite`) — real probing needs the
engine + sandbox, which the background job wires; this module owns the
quarantine gate, the scorecard store + persistence, and the battery orchestra-
tion. Pure/fail-open: any error yields a neutral, non-quarantining result so
routing is never blocked. Persistence mirrors `quota_plan.py` (best-effort
`llm_settings` KV, rehydrate on boot) — no new migration.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

# Re-probe a pair this long after its last probe (rolling freshness, ~monthly).
_REPROBE_AFTER_S = 30 * 86_400.0


@dataclass
class Scorecard:
    """Per-`(CanonicalId, provider)` measured stats — the §2.6 quality basis."""
    json_reliability: float = 1.0     # 0..1 schema-valid rate over the battery
    quality_prior: float = 1.0        # 0..1 instruction-following prior
    coder_prior: float = 1.0          # 0..1 sandbox-compiled code-smoke rate
    context_effective: float = 1.0    # 0..1 needle recall at 25/50/90%
    ttft_s: float = 0.0               # measured time-to-first-token
    tps: float = 0.0                  # measured tokens/sec
    capabilities: dict = field(default_factory=dict)  # streaming/system/stop/tools
    probed_at: float = 0.0            # epoch of the last probe (0 = never)
    error_signature: str = ""         # provider error fingerprint at probe time


@dataclass
class ProbeSuite:
    """The battery's probes, injected so the module makes no real model calls.
    Each returns the metric for the `(cid_key, provider)` pair; any may be None
    to leave that stat at its neutral default. All async."""
    schema: Callable[[str, str], Awaitable[float]] | None = None
    instruction: Callable[[str, str], Awaitable[float]] | None = None
    code_smoke: Callable[[str, str], Awaitable[float]] | None = None
    needle: Callable[[str, str], Awaitable[float]] | None = None
    timed: Callable[[str, str], Awaitable[tuple]] | None = None
    conformance: Callable[[str, str], Awaitable[dict]] | None = None


def _clamp(v, default=1.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


class Gauntlet:
    """Quarantine gate + scorecard store for onboarding new model/provider pairs.
    In-process, injectable clock, fail-open."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._cards: dict[str, Scorecard] = {}
        self._now = now or time.time

    @staticmethod
    def _key(cid_key: str, provider: str) -> str:
        return f"{cid_key}@{(provider or '').strip().lower()}"

    # ── quarantine gate ───────────────────────────────────────────────────
    def is_probed(self, cid_key: str, provider: str) -> bool:
        card = self._cards.get(self._key(cid_key, provider))
        return card is not None and card.probed_at > 0

    def is_quarantined(self, cid_key: str, provider: str) -> bool:
        """A pair with no passing probe yet → not selectable above T3."""
        return not self.is_probed(cid_key, provider)

    def scorecard(self, cid_key: str, provider: str) -> Scorecard | None:
        return self._cards.get(self._key(cid_key, provider))

    def record(self, cid_key: str, provider: str, card: Scorecard) -> None:
        self._cards[self._key(cid_key, provider)] = card

    # ── re-probe triggers ─────────────────────────────────────────────────
    def needs_reprobe(self, cid_key: str, provider: str, *,
                      error_signature: str | None = None,
                      reprobe_after_s: float = _REPROBE_AFTER_S) -> bool:
        card = self._cards.get(self._key(cid_key, provider))
        if card is None or card.probed_at <= 0:
            return True                                   # never probed
        if self._now() - card.probed_at >= reprobe_after_s:
            return True                                   # stale (rolling)
        if error_signature and error_signature != card.error_signature:
            return True                                   # provider changed shape
        return False

    def note_error_signature(self, cid_key: str, provider: str,
                             signature: str) -> None:
        """Record a fresh provider error fingerprint; a change flags a re-probe
        (a model-update / API change often shows up as a new error shape first)."""
        card = self._cards.get(self._key(cid_key, provider))
        if card is not None and signature and signature != card.error_signature:
            # Invalidate the probe so the gate re-runs the battery.
            card.error_signature = signature
            card.probed_at = 0.0

    # ── the probe battery ─────────────────────────────────────────────────
    async def run_battery(self, cid_key: str, provider: str, *,
                          probes: ProbeSuite,
                          error_signature: str = "") -> Scorecard:
        """Run the injected probes and BUILD the scorecard (results ARE the
        scorecard). A probe that's absent or errors leaves its stat neutral —
        the gauntlet never fails a candidate for a probe HARNESS bug, only for a
        measured deficiency. Records + returns the card."""
        card = Scorecard(probed_at=self._now(), error_signature=error_signature)

        async def _one(fn, setter):
            if fn is None:
                return
            try:
                setter(await fn(cid_key, provider))
            except Exception as exc:  # noqa: BLE001 — a probe error is neutral
                log.info("gauntlet probe error (%s@%s): %s",
                         cid_key, provider, exc)

        await _one(probes.schema,
                   lambda v: setattr(card, "json_reliability", _clamp(v)))
        await _one(probes.instruction,
                   lambda v: setattr(card, "quality_prior", _clamp(v)))
        await _one(probes.code_smoke,
                   lambda v: setattr(card, "coder_prior", _clamp(v)))
        await _one(probes.needle,
                   lambda v: setattr(card, "context_effective", _clamp(v)))

        def _set_timed(v):
            try:
                ttft, tps = v
                card.ttft_s = max(0.0, float(ttft))
                card.tps = max(0.0, float(tps))
            except Exception:  # noqa: BLE001
                pass
        await _one(probes.timed, _set_timed)
        await _one(probes.conformance,
                   lambda v: setattr(card, "capabilities", dict(v or {})))

        self.record(cid_key, provider, card)
        return card

    def snapshot(self) -> list[dict]:
        out = []
        for k, card in self._cards.items():
            cid_key, _, provider = k.rpartition("@")
            row = {"cid": cid_key, "provider": provider}
            row.update(asdict(card))
            out.append(row)
        return out

    def clear(self) -> None:
        self._cards.clear()


# --------------------------------------------------------------------------- #
_gauntlet = Gauntlet()


def gauntlet() -> Gauntlet:
    return _gauntlet


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.routing, "gauntlet", False))
    except Exception:  # noqa: BLE001
        return False


def is_quarantined(cid_key: str, provider: str) -> bool:
    """Router gate — True when the pair is unproven AND the gauntlet is on."""
    if not enabled():
        return False
    try:
        return _gauntlet.is_quarantined(cid_key, provider)
    except Exception:  # noqa: BLE001
        return False


# ── best-effort persistence (mirrors quota_plan.py; reuses llm_settings KV) ──
_SETTING_KEY = "gauntlet_scorecards"


def _fire(coro) -> None:
    import asyncio
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()


async def persist() -> None:
    if not enabled():
        return
    try:
        import json

        from sqlalchemy.dialects.postgresql import insert
        from storage.db import get_session_factory
        from storage.models import LLMSetting
        factory = get_session_factory()
        if factory is None:
            return
        blob = json.dumps(_gauntlet.snapshot(), separators=(",", ":"))
        async with factory() as session:
            stmt = insert(LLMSetting).values(key=_SETTING_KEY, value=blob) \
                .on_conflict_do_update(index_elements=["key"],
                                       set_={"value": blob})
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("gauntlet persist failed: %s", exc)


async def rehydrate() -> int:
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
            cid = r.pop("cid", ""); prov = r.pop("provider", "")
            if not cid:
                continue
            _gauntlet.record(cid, prov, Scorecard(**r))
            loaded += 1
        return loaded
    except Exception as exc:  # noqa: BLE001
        log.debug("gauntlet rehydrate failed: %s", exc)
        return 0


def reset_for_tests() -> None:
    _gauntlet.clear()


__all__ = ["Scorecard", "ProbeSuite", "Gauntlet", "gauntlet", "enabled",
           "is_quarantined", "persist", "rehydrate", "reset_for_tests"]
