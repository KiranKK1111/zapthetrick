"""
HR intent classification + fact-based Negotiation_Strategy
(live-conversational-intelligence R43).

`classify_hr_intent` maps an HR-phase question to a structured intent
(salary / notice-period / counter-offer / why-join / why-leaving / benefits).
`negotiation_strategy` produces a FACT-BASED talking strategy grounded in the
candidate profile + org/role + (optional) market data — never a manipulation
script. An explicit no-manipulation guard strips coercive/deceptive tactics and
an unrealistic-ask risk flag warns when an ask is far outside a sane band.
Deterministic + fail-open. Advisory only; folded into the existing answer call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core import lexicons

# Intent identifiers.
SALARY = "salary"
LOW_OFFER = "low_offer"
VALUE_JUSTIFICATION = "value_justification"
FINAL_OFFER = "final_offer"
NOTICE_PERIOD = "notice_period"
COUNTER_OFFER = "counter_offer"
WHY_JOIN = "why_join"
WHY_LEAVING = "why_leaving"
BENEFITS = "benefits"
OTHER = "other"

_INTENT_CUES = lexicons.LIVE_NEGOTIATE_INTENT_CUES

# Coercive / deceptive phrasings the guard must never emit.
_MANIPULATION = lexicons.LIVE_NEGOTIATE_MANIPULATION

# --------------------------------------------------------------------------- #
# Coarse, EXPLICITLY-APPROXIMATE salary reference (BandSpecific.md lines 146-161,
# India CTC). This is deliberately a starting anchor, NOT authoritative data:
# real numbers vary widely by company, city, and equity and go stale fast. The
# strategic, no-hardcoded-number handling in `negotiation_strategy` is the
# primary design; this table is CONSULTED ONLY when explicitly opted in
# (`use_reference=True`) AND no live market band was supplied, and every point
# it produces carries the caveat. Keyed by seniority-band slug → (low, high) LPA.
# A deployment can override this dict for its own market.
# --------------------------------------------------------------------------- #
APPROX_SALARY_BANDS_LPA: dict[str, tuple[float, float]] = {
    "intern": (3.0, 8.0),
    "fresher": (4.0, 10.0),
    "junior": (6.0, 14.0),
    "mid": (8.0, 20.0),
    "senior": (18.0, 40.0),
    "lead": (30.0, 70.0),
    "principal": (60.0, 150.0),
    "distinguished": (100.0, 300.0),
}

SALARY_REFERENCE_CAVEAT = (
    "approximate India-market reference — varies widely by company, city, and equity, "
    "and dates quickly; treat as a rough starting anchor and verify against current data")


def approx_band_range(band_slug: str | None) -> tuple[float, float] | None:
    """Coarse approximate (low, high) LPA for a seniority-band slug, or None if
    unknown. EXPLICITLY approximate — see `SALARY_REFERENCE_CAVEAT`. Never raises."""
    try:
        return APPROX_SALARY_BANDS_LPA.get((band_slug or "").strip().lower())
    except Exception:  # noqa: BLE001
        return None


def classify_hr_intent(question: str) -> str:
    """Classify an HR question into a structured intent. Never raises."""
    try:
        t = (question or "").lower()
        if not t.strip():
            return OTHER
        for intent, cues in _INTENT_CUES.items():
            if any(c in t for c in cues):
                return intent
        return OTHER
    except Exception:  # noqa: BLE001
        return OTHER


@dataclass
class NegotiationStrategy:
    intent: str = OTHER
    points: list[str] = field(default_factory=list)
    risk_flag: str = ""   # e.g. "unrealistic_ask" — advisory warning

    def to_dict(self) -> dict:
        return {"intent": self.intent, "points": self.points, "risk_flag": self.risk_flag}


def _no_manipulation(points: list[str]) -> list[str]:
    """Strip any point that reads as coercive/deceptive (no-manipulation guard)."""
    clean = []
    for p in points:
        low = (p or "").lower()
        if any(bad in low for bad in _MANIPULATION):
            continue
        clean.append(p)
    return clean


def negotiation_strategy(
    question: str,
    *,
    strengths: list[str] | None = None,
    market_low: float | None = None,
    market_high: float | None = None,
    ask: float | None = None,
    candidate_stated: dict | None = None,
    interviewer_signal: str | None = None,
    seniority_band: str | None = None,
    use_reference: bool = False,
) -> NegotiationStrategy:
    """Build a fact-based, non-manipulative negotiation strategy. Never raises.

    `candidate_stated` carries what the candidate ALREADY said out loud (from the
    world-model commitments), e.g. {"salary": "25 LPA", "competing_offer": "yes",
    "notice_period": "30 days"}. `interviewer_signal` carries a pushback signal
    (e.g. "interviewer_thinks_high"). When present, the strategy reacts to what
    was actually said instead of giving generic advice."""
    s = NegotiationStrategy(intent=classify_hr_intent(question))
    try:
        pts: list[str] = []
        stated = candidate_stated or {}
        if s.intent == SALARY:
            stated_salary = stated.get("salary")
            if stated_salary:
                # The candidate already committed to a number → hold + justify it,
                # do NOT re-anchor lower.
                pts.append(f"You already stated {stated_salary} — hold that figure and "
                           "justify it; don't undercut yourself by re-anchoring lower.")
                if interviewer_signal == "interviewer_thinks_high":
                    pts.append("The interviewer signalled it's high — justify with market data "
                               "and your concrete value, or offer a tight range around it "
                               "rather than dropping sharply.")
                elif interviewer_signal == "interviewer_thinks_low":
                    pts.append("The interviewer hinted you could ask for more — it's reasonable "
                               "to revise upward toward the market band.")
            if market_low is not None and market_high is not None:
                pts.append(f"Anchor on the market band ({market_low:g}-{market_high:g}) for the role.")
            elif use_reference and not stated_salary:
                # No live market band supplied and explicitly opted in → offer a
                # COARSE, clearly-approximate reference the candidate can sanity-
                # check against, never presented as authoritative.
                ref = approx_band_range(seniority_band)
                if ref is not None:
                    pts.append(f"As a rough reference (~{ref[0]:g}-{ref[1]:g} LPA for this band; "
                               f"{SALARY_REFERENCE_CAVEAT}), anchor on a researched range and "
                               "confirm against up-to-date market data before committing.")
            if strengths:
                pts.append("Justify with concrete value: " + ", ".join(strengths[:5]) + ".")
            if not stated_salary:
                pts.append("State a researched range, not a single number; stay collaborative.")
            if stated.get("competing_offer") == "yes":
                pts.append("You mentioned a competing offer — reference it factually as leverage; "
                           "never invent or inflate one.")
            # Unrealistic-ask risk flag (advisory).
            if ask is not None and market_high is not None and ask > market_high * 1.5:
                s.risk_flag = "unrealistic_ask"
                pts.append("Note: this ask is well above the market band — be ready to justify or adjust.")
        elif s.intent == COUNTER_OFFER:
            if stated.get("competing_offer") == "yes":
                pts.append("You have a real competing offer — state it honestly; focus on fit + growth.")
            else:
                pts.append("Be honest about competing offers only if real; never bluff a fake offer.")
                pts.append("Focus on fit and growth, not just the number.")
        elif s.intent == NOTICE_PERIOD:
            stated_notice = stated.get("notice_period")
            if stated_notice:
                pts.append(f"You stated a notice period of {stated_notice} — keep it consistent; "
                           "offer realistic flexibility if any.")
            else:
                pts.append("State your actual notice period; offer realistic flexibility if any.")
        elif s.intent == LOW_OFFER:
            # Acknowledge → reinforce value → counter politely (fact-based).
            pts.append("Acknowledge the offer, then reinforce your value with concrete impact.")
            if strengths:
                pts.append("Anchor on your strengths: " + ", ".join(strengths[:4]) + ".")
            pts.append("Counter politely toward your researched market range; ask if there's "
                       "flexibility in the overall package (base, bonus, equity).")
        elif s.intent == VALUE_JUSTIFICATION:
            pts.append("Justify with measurable impact and scope, not seniority claims.")
            if strengths:
                pts.append("Lead with proof points: " + ", ".join(strengths[:4]) + ".")
        elif s.intent == FINAL_OFFER:
            pts.append("Treat a 'final offer' respectfully: restate fit and interest, then ask "
                       "once about non-cash levers (equity, sign-on, review timeline) before deciding.")
            pts.append("Do not bluff or issue ultimatums; decide on real fit + value.")
        elif s.intent == WHY_JOIN:
            pts.append("Tie genuine motivation to the role/company specifics and your strengths.")
        elif s.intent == WHY_LEAVING:
            pts.append("Frame the move around growth, not negativity about the current employer.")
        elif s.intent == BENEFITS:
            pts.append("Clarify the full package (base, equity, bonus) before negotiating components.")
        s.points = _no_manipulation(pts)
        return s
    except Exception:  # noqa: BLE001
        return s


def directive(strategy: NegotiationStrategy) -> str:
    """A fact-based, no-manipulation directive for the answer call."""
    try:
        if not strategy.points:
            return ""
        head = "HR/negotiation guidance (be factual, never manipulative): "
        return head + " ".join(strategy.points)
    except Exception:  # noqa: BLE001
        return ""
