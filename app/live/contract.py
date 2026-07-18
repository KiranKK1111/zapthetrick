"""Conversation Contract Engine (roadmap Phase 2 #23).

A per-session contract — role, goal, style, and a max spoken length — established
once and validated against every suggested answer, so the assistant doesn't drift
over a long interview (rambling, wrong register). Deterministic + fail-open;
per-session registry mirrors phase_tracker.py / state_machine.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Rough spoken-delivery rate (characters per second) for turning a max-seconds
# budget into a character budget. ~150 wpm ≈ ~13 cps at ~5.2 chars/word.
_CPS = 13.0


@dataclass(frozen=True)
class Contract:
    role: str = "candidate"
    goal: str = "interview"
    style: str = "professional"
    max_answer_seconds: int = 60

    def max_chars(self, *, tolerance: float = 1.2) -> int:
        return int(self.max_answer_seconds * _CPS * tolerance)


@dataclass
class ContractCheck:
    ok: bool
    violations: list[str] = field(default_factory=list)


# ── Adaptive length / multi-level answers (roadmap Phase 2 #21 / 2C-21) ─────
# Three spoken-length tiers so the same question can be answered at 10s (a crisp
# headline), 30s (standard), or 60s (deep). The contract's max_answer_seconds
# picks the default tier; the others are offered so the FE / candidate can ask
# for "shorter" or "go deeper" without re-asking.
_VARIANTS = {"brief": 10, "standard": 30, "deep": 60}


def variant_budgets() -> dict[str, int]:
    """{tier: max_chars} for the 10s / 30s / 60s answer variants."""
    return {name: int(secs * _CPS * 1.2) for name, secs in _VARIANTS.items()}


def variant_for_seconds(max_seconds: int) -> str:
    """The tier a max-seconds budget maps to (nearest, not exceeding when
    possible)."""
    try:
        s = int(max_seconds)
        if s <= 15:
            return "brief"
        if s <= 40:
            return "standard"
        return "deep"
    except Exception:  # noqa: BLE001
        return "standard"


def length_directive(contract: "Contract") -> str:
    """Directive telling the answer which tier to target and to offer a shorter
    take. Never raises."""
    try:
        tier = variant_for_seconds(contract.max_answer_seconds)
        shape = {
            "brief": "a crisp headline answer (~10s) — one or two sentences",
            "standard": "a standard answer (~30s) — the point plus one example",
            "deep": "a thorough answer (~60s) — the point, mechanism, and a trade-off",
        }[tier]
        return f"Target {shape}; be ready to expand or compress if asked."
    except Exception:  # noqa: BLE001
        return ""


def derive_contract(
    *,
    phase: str = "",
    style: str = "professional",
    max_answer_seconds: int | None = None,
) -> Contract:
    """Build a contract from early session signals. Later phases (e.g. HR) get a
    tighter default speaking window unless overridden."""
    secs = max_answer_seconds
    if secs is None:
        secs = 45 if phase in ("hr", "closing", "behavioral") else 60
    return Contract(style=style, max_answer_seconds=secs)


def validate(answer: str, contract: Contract) -> ContractCheck:
    """Check a suggested answer against the contract. Currently enforces the
    spoken-length budget (the most common drift); returns typed violations."""
    try:
        violations: list[str] = []
        if answer and len(answer) > contract.max_chars():
            violations.append("too_long")
        return ContractCheck(ok=not violations, violations=violations)
    except Exception:  # noqa: BLE001
        return ContractCheck(ok=True)


_contracts: dict[str, Contract] = {}


def get_contract(session_id: str) -> Contract | None:
    return _contracts.get(session_id)


def set_contract(session_id: str, contract: Contract) -> Contract:
    _contracts[session_id] = contract
    return contract


def ensure_contract(session_id: str, **kwargs) -> Contract:
    c = _contracts.get(session_id)
    if c is None:
        c = derive_contract(**kwargs)
        _contracts[session_id] = c
    return c


def forget_session(session_id: str) -> None:
    _contracts.pop(session_id, None)


__all__ = [
    "Contract", "ContractCheck", "derive_contract", "validate",
    "get_contract", "set_contract", "ensure_contract", "forget_session",
]
