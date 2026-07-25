"""Band contracts precompute (vNext §4.3, Stage 6 Component K).

The seniority BAND, career TRACK, and answer GUIDANCE for a candidate don't
change turn-to-turn — they're a function of the resume + the target role. §4.3
computes them ONCE at profile-land and PINS them, so every live answer reads a
cached contract instead of re-deriving calibration each turn. Each band also
carries a **structural contract** — the shape a good answer at that band takes
(depth, ownership language, the sections it should hit, what to include / avoid)
— so an intern's answer stays a crisp concrete example while a principal's frames
system + org tradeoffs.

`calibration.py` already computes the band/track/guidance (`build_calibration`);
this module owns the per-band STRUCTURAL contracts (data) + the pin cache. The
caller passes the computed bands in (no coupling to the heavy calibration build).
Pure + fail-open. Flag-gated (`live.band_contracts`, default OFF → today's
per-turn `calibration_directive`).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

_LOCK = threading.RLock()


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "band_contracts", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class BandContract:
    band: str
    depth: str                 # how deep an answer should go
    ownership: str             # the ownership language that fits the band
    max_seconds: int           # spoken-length ceiling
    sections: tuple[str, ...]  # the beats a good answer hits
    must_include: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"band": self.band, "depth": self.depth,
                "ownership": self.ownership, "max_seconds": self.max_seconds,
                "sections": list(self.sections),
                "must_include": list(self.must_include),
                "avoid": list(self.avoid)}


# Per-band structural contracts (the YAML-equivalent data table). A YAML asset can
# override these later; this is the canonical form.
_BAND_CONTRACTS: dict[str, BandContract] = {
    "intern": BandContract(
        "intern", depth="one concrete example, plainly told", ownership="contributed to, with guidance",
        max_seconds=20, sections=("what I did", "what I learned"),
        avoid=("system design claims", "team-ownership language")),
    "fresher": BandContract(
        "fresher", depth="a focused example + the takeaway", ownership="built, with guidance",
        max_seconds=25, sections=("the task", "my approach", "the result"),
        avoid=("architecture ownership", "leading a team")),
    "junior": BandContract(
        "junior", depth="approach + outcome on a scoped task", ownership="owned a scoped piece",
        max_seconds=30, sections=("context", "what I built", "the outcome"),
        avoid=("org-level impact", "setting technical direction")),
    "mid": BandContract(
        "mid", depth="approach + a tradeoff + a measurable outcome", ownership="owned / delivered",
        max_seconds=35, sections=("context", "approach", "a tradeoff", "impact"),
        must_include=("a concrete metric or outcome",)),
    "senior": BandContract(
        "senior", depth="problem framing + tradeoffs + impact", ownership="led / drove",
        max_seconds=45, sections=("the problem", "approach & tradeoffs", "impact", "what I'd do differently"),
        must_include=("a tradeoff you weighed", "a measurable result")),
    "lead": BandContract(
        "lead", depth="technical direction + cross-team impact", ownership="set direction for / aligned",
        max_seconds=50, sections=("the situation", "the technical direction I set", "how I aligned people", "the business impact"),
        must_include=("an influence-beyond-code example",)),
    "principal": BandContract(
        "principal", depth="system-level framing + org & business impact", ownership="owned the strategy for",
        max_seconds=60, sections=("problem framing", "architecture & tradeoffs", "org / business impact", "the risks I managed"),
        avoid=("low-level implementation trivia unless asked",),
        must_include=("a system-wide or org-wide outcome",)),
    "distinguished": BandContract(
        "distinguished", depth="strategy, org influence, long-horizon bets", ownership="shaped the direction of",
        max_seconds=60, sections=("the strategic problem", "the bet I made", "how I moved the org", "long-term impact"),
        avoid=("day-to-day coding detail unless asked",)),
}
_DEFAULT_BAND = "mid"


def structural_contract(band: str) -> BandContract:
    """The structural answer contract for a band slug (falls back to 'mid' for an
    unknown band). Never raises."""
    try:
        return _BAND_CONTRACTS.get((band or "").strip().lower(),
                                   _BAND_CONTRACTS[_DEFAULT_BAND])
    except Exception:  # noqa: BLE001
        return _BAND_CONTRACTS[_DEFAULT_BAND]


# --------------------------------------------------------------------------- #
# Pin cache — compute once at profile-land, reuse every turn
# --------------------------------------------------------------------------- #
@dataclass
class PinnedContract:
    key: str
    real_band: str
    target_band: str = ""
    track: str = ""
    guidance: str = ""
    structural: BandContract = field(
        default_factory=lambda: _BAND_CONTRACTS[_DEFAULT_BAND])
    pinned_at: float = 0.0

    def directive(self) -> str:
        """A compact structural answer directive from the pinned contract — the
        band's shape, framed toward the target band where genuine."""
        try:
            sc = self.structural
            bits = [f"Answer at a {sc.band}-level: {sc.depth}."]
            if self.target_band and self.target_band != self.real_band:
                bits.append(f"Frame toward a {self.target_band} role where you "
                            "have genuine depth, but never overclaim seniority.")
            if sc.sections:
                bits.append("Hit: " + " → ".join(sc.sections) + ".")
            if sc.must_include:
                bits.append("Include " + ", ".join(sc.must_include) + ".")
            if sc.avoid:
                bits.append("Avoid " + ", ".join(sc.avoid) + ".")
            if self.track:
                bits.append(f"This is a {self.track} interview.")
            if self.guidance:
                bits.append(self.guidance)
            return " ".join(bits)
        except Exception:  # noqa: BLE001
            return self.guidance or ""


_pins: dict[str, PinnedContract] = {}


def pin(key: str, *, real_band: str, target_band: str = "", track: str = "",
        guidance: str = "", now: float | None = None) -> PinnedContract:
    """Precompute + pin the band contract for a profile/session `key` (called ONCE
    at profile-land). Reuses the structural contract for the REAL band. Never
    raises."""
    ts = time.time() if now is None else now
    pc = PinnedContract(
        key=key, real_band=(real_band or _DEFAULT_BAND).strip().lower(),
        target_band=(target_band or "").strip().lower(), track=track,
        guidance=guidance, structural=structural_contract(real_band),
        pinned_at=ts)
    with _LOCK:
        _pins[key] = pc
    return pc


def get_pinned(key: str) -> PinnedContract | None:
    with _LOCK:
        return _pins.get(key)


def forget(key: str) -> None:
    with _LOCK:
        _pins.pop(key, None)


def reset_for_tests() -> None:
    with _LOCK:
        _pins.clear()


__all__ = ["enabled", "BandContract", "structural_contract", "PinnedContract",
           "pin", "get_pinned", "forget", "reset_for_tests"]
