"""Spend accounting and ceilings for the metered realtime engine (§Budget).

Cloud realtime is billed per audio token, and published per-minute figures vary
widely between sources and measured sessions. Trusting any single headline
number would be a mistake, so this module **meters actual usage** reported by
the engine (`UsageDelta`) and converts it with prices that live in config, not
in code.

Guarantees this module provides:

* Cumulative metered spend never exceeds the configured ceiling (Property 8).
* Under default config — `engine: staged`, empty `realtime_model`, zero
  ceilings — metered spend is exactly zero, because a zero ceiling reads as "no
  budget remaining" and the policy refuses to select realtime at all.
* Ceilings are checked *during* a session, not after it, so a runaway is stopped
  rather than discovered.

The ledger is a small JSON file with day and month buckets only — no PII, no
transcript, nothing that needs a migration. Every read/write is fail-open: a
corrupt or unwritable ledger degrades to "no spend recorded" rather than
breaking a conversation, and the empty-model interlock still holds.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger("zapthetrick.voice.budget")

_LEDGER_PATH = os.path.join("data", "voice_spend.json")
_lock = threading.Lock()


def _cfg():
    from app.core.config_loader import cfg
    return cfg.voice


def _budget_cfg():
    b = getattr(_cfg(), "budget", None)
    if b is None:                       # pre-upgrade config with no block
        from app.core.config_loader import VoiceBudgetConfig
        return VoiceBudgetConfig()
    return b


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _this_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


@dataclass
class Spend:
    """What has been spent, in dollars, for the current day and month."""
    day: float = 0.0
    month: float = 0.0


def _load() -> dict:
    try:
        with open(_LEDGER_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001 — a corrupt ledger must not break voice
        log.warning("voice spend ledger unreadable — treating as empty",
                    exc_info=True)
        return {}


def _save(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_LEDGER_PATH) or ".", exist_ok=True)
        tmp = _LEDGER_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, _LEDGER_PATH)   # atomic — never a half-written ledger
    except Exception:  # noqa: BLE001
        log.warning("voice spend ledger unwritable — spend not persisted",
                    exc_info=True)


def price(input_tokens: int = 0, output_tokens: int = 0,
          cached_tokens: int = 0) -> float:
    """Convert metered tokens into dollars using the configured prices."""
    b = _budget_cfg()
    return (
        (max(0, input_tokens) / 1e6) * float(b.audio_input_per_mtok)
        + (max(0, output_tokens) / 1e6) * float(b.audio_output_per_mtok)
        + (max(0, cached_tokens) / 1e6) * float(b.cached_input_per_mtok)
    )


def spent() -> Spend:
    """Spend so far today and this month. Buckets that are not the current
    day/month read as zero, so rollover needs no cron."""
    with _lock:
        data = _load()
    d = data.get("day") or {}
    m = data.get("month") or {}
    return Spend(
        day=float(d.get(_today(), 0.0) or 0.0),
        month=float(m.get(_this_month(), 0.0) or 0.0),
    )


def record(usd: float) -> Spend:
    """Add `usd` to today's and this month's buckets. Returns the new totals."""
    amount = max(0.0, float(usd or 0.0))
    if amount <= 0.0:
        return spent()
    with _lock:
        data = _load()
        day = data.setdefault("day", {})
        month = data.setdefault("month", {})
        dk, mk = _today(), _this_month()
        day[dk] = round(float(day.get(dk, 0.0) or 0.0) + amount, 6)
        month[mk] = round(float(month.get(mk, 0.0) or 0.0) + amount, 6)
        # Keep the file small: only the last ~60 days / 24 months matter.
        if len(day) > 60:
            for k in sorted(day)[:-60]:
                day.pop(k, None)
        if len(month) > 24:
            for k in sorted(month)[:-24]:
                month.pop(k, None)
        _save(data)
        return Spend(day=day[dk], month=month[mk])


def remaining() -> float:
    """Dollars left before the tightest configured ceiling is reached.

    A ceiling of 0 means "no budget" and returns 0.0 — NOT "unlimited". That
    asymmetry is deliberate: the safe reading of an unconfigured budget is that
    nothing may be spent (Requirement 9.1).
    """
    b = _budget_cfg()
    s = spent()
    day_left = max(0.0, float(b.daily_usd) - s.day)
    month_left = max(0.0, float(b.monthly_usd) - s.month)
    return min(day_left, month_left)


def ceiling_fraction() -> float:
    """How much of the tightest ceiling is consumed, in [0, 1]. Returns 1.0 when
    no ceiling is configured (nothing may be spent, so any spend is 'full')."""
    b = _budget_cfg()
    s = spent()
    fracs = []
    if float(b.daily_usd) > 0:
        fracs.append(s.day / float(b.daily_usd))
    if float(b.monthly_usd) > 0:
        fracs.append(s.month / float(b.monthly_usd))
    if not fracs:
        return 1.0
    return max(0.0, min(1.0, max(fracs)))


def should_warn() -> bool:
    """True once spend crosses `warn_at` of a ceiling but is not yet exhausted
    — the point at which the UI tells the user (Requirement 9.3)."""
    b = _budget_cfg()
    if float(b.daily_usd) <= 0 and float(b.monthly_usd) <= 0:
        return False
    f = ceiling_fraction()
    return float(b.warn_at) <= f < 1.0


def session_reserve() -> float:
    """Dollars a session should have available before realtime is selected.

    Opening a realtime session that can only run for seconds is worse than not
    opening one: the user gets a switch notice mid-sentence. Reserve one minute
    of two-way audio at the configured prices — enough to be worth starting.
    """
    # ~1 min of audio in and out. Realtime audio runs on the order of 10 tokens
    # per 100 ms in each direction; this is an order-of-magnitude reserve, not a
    # precise quote, and it only gates whether a session is worth opening.
    return price(input_tokens=6_000, output_tokens=6_000)


def can_open_session() -> tuple[bool, str]:
    """Whether a realtime session may be opened right now."""
    b = _budget_cfg()
    if float(b.daily_usd) <= 0 and float(b.monthly_usd) <= 0:
        return False, "no voice budget configured"
    left = remaining()
    if left <= 0:
        return False, "voice spend ceiling reached"
    reserve = session_reserve()
    if left < reserve:
        return False, (f"remaining budget ${left:.2f} below the "
                       f"${reserve:.2f} session reserve")
    return True, ""


def session_seconds_cap() -> float:
    """Hard wall-clock cap for one realtime session, in seconds."""
    return max(0.0, float(_budget_cfg().session_minutes) * 60.0)


class SessionMeter:
    """Per-session accumulator. The runner feeds it `UsageDelta`s; it answers
    'may this session continue' and persists spend as it goes.

    Spend is written through on every update rather than at session end, so a
    crashed process cannot lose the record of money already spent.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.usd = 0.0
        self._persisted = 0.0

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0,
            cached_tokens: int = 0) -> float:
        """Meter a usage delta. Returns total session spend in dollars."""
        self.input_tokens += max(0, int(input_tokens or 0))
        self.output_tokens += max(0, int(output_tokens or 0))
        self.cached_tokens += max(0, int(cached_tokens or 0))
        self.usd = price(self.input_tokens, self.output_tokens,
                         self.cached_tokens)
        delta = self.usd - self._persisted
        if delta > 0:
            record(delta)
            self._persisted = self.usd
        return self.usd

    def exhausted(self) -> bool:
        """True once the global ceiling is reached. The runner lets the current
        turn finish, then switches engine (Requirement 9.4)."""
        return remaining() <= 0.0


__all__ = [
    "Spend", "price", "spent", "record", "remaining", "ceiling_fraction",
    "should_warn", "session_reserve", "can_open_session",
    "session_seconds_cap", "SessionMeter",
]
