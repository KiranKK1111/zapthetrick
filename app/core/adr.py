"""Architecture Decision Record (ADR) engine (roadmap Phase 8A #11).

A lightweight, queryable registry of architectural decisions — each with its
rationale, trade-offs, status, and supersession chain — so the *why* behind the
architecture is recorded and inspectable instead of tribal knowledge. Composes
with the Phase 0 governance guardrails: the import-baseline / allowlist diffs are
ADR seeds; this formalizes them.

Deterministic + fail-open. Seeded with the real decisions made while building the
reliability/guardrail layers so it isn't an empty shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROPOSED = "proposed"
ACCEPTED = "accepted"
SUPERSEDED = "superseded"
DEPRECATED = "deprecated"
_STATUSES = {PROPOSED, ACCEPTED, SUPERSEDED, DEPRECATED}


@dataclass
class ADR:
    id: str
    title: str
    rationale: str
    status: str = ACCEPTED
    tradeoffs: str = ""
    date: str = ""                    # ISO date (caller-supplied; no wall-clock here)
    supersedes: str | None = None     # id this decision replaces
    superseded_by: str | None = None  # set when a later ADR replaces this one


_adrs: dict[str, ADR] = {}


def record(adr: ADR) -> ADR:
    """Register (or replace) an ADR. If it supersedes another, mark that one."""
    try:
        if adr.status not in _STATUSES:
            adr = ADR(**{**adr.__dict__, "status": PROPOSED})
        _adrs[adr.id] = adr
        if adr.supersedes and adr.supersedes in _adrs:
            old = _adrs[adr.supersedes]
            _adrs[adr.supersedes] = ADR(**{**old.__dict__,
                                           "status": SUPERSEDED,
                                           "superseded_by": adr.id})
    except Exception:  # noqa: BLE001 — recording a decision must never crash
        pass
    return adr


def get(adr_id: str) -> ADR | None:
    return _adrs.get(adr_id)


def all_adrs(*, status: str | None = None) -> list[ADR]:
    items = sorted(_adrs.values(), key=lambda a: a.id)
    return [a for a in items if status is None or a.status == status]


def active() -> list[ADR]:
    """Decisions still in force (accepted, not superseded/deprecated)."""
    return [a for a in all_adrs() if a.status == ACCEPTED]


def deprecate(adr_id: str) -> None:
    a = _adrs.get(adr_id)
    if a:
        _adrs[adr_id] = ADR(**{**a.__dict__, "status": DEPRECATED})


def reset() -> None:
    _adrs.clear()
    _seed()


def _seed() -> None:
    """The real architectural decisions from the reliability/guardrail build-out."""
    for a in _SEED:
        _adrs.setdefault(a.id, a)


_SEED: list[ADR] = [
    ADR("ADR-0001", "Architecture guardrails as CI fitness functions",
        rationale="Anti-pattern rules that live only in docs get violated; encode "
                  "them as static AST tests (tests/architecture/) that fail CI on "
                  "new violations — green-by-construction from current reality.",
        tradeoffs="Allowlists need occasional intentional updates (which are the ADR seeds).",
        date="2026-07-11"),
    ADR("ADR-0002", "obs/ is the home for cross-cutting reliability primitives",
        rationale="failure_taxonomy, recovery, failure_prediction, failure_kb, replay "
                  "compose cleanly and are consumed by many packages; centralizing "
                  "them in obs/ avoids a new top-level package (governance rule #20).",
        tradeoffs="obs/ grows; mitigated by clear single-responsibility modules.",
        date="2026-07-11"),
    ADR("ADR-0003", "Import boundaries frozen as a reviewed baseline",
        rationale="New cross-package coupling must be a conscious decision; the "
                  "import-baseline.json diff records each intentional edge (e.g. "
                  "llm→obs, response_arch→core) as an ADR.",
        tradeoffs="A deliberate edit is needed to add coupling — that friction is the point.",
        date="2026-07-11"),
    ADR("ADR-0004", "Reliability layers compose across phases, not silos",
        rationale="Phase 1 taxonomy names failures, Phase 4 recovery plans them, "
                  "Phase 7 failure_kb learns which recovery works — each builds on "
                  "the prior instead of a parallel mechanism.",
        tradeoffs="Cross-phase dependencies; kept unidirectional and fail-open.",
        date="2026-07-12"),
]

_seed()


__all__ = [
    "ADR", "PROPOSED", "ACCEPTED", "SUPERSEDED", "DEPRECATED",
    "record", "get", "all_adrs", "active", "deprecate", "reset",
]
