"""Aggregate confidence + gating (evaluation-and-reliability R3/R4, tasks 4.2/5.3).

Pins Properties 4 & 5: each subsystem emits a well-formed signal (neutral on
missing), aggregation reuses the trust shape + bands correctly, and gating maps
high→proceed / low→clarifier / mid→judgment with no new ask path.
"""
from __future__ import annotations

from app.chat.trust import ConfidenceResult, ConfidenceSignals, confidence_band
from app.quality import confidence as qc


# ── per-subsystem emission (Property 4) ──────────────────────────────────────
def test_from_routing_emits_signal_and_missing_is_none():
    s = qc.from_routing("trivial")
    assert s is not None and s.source == "routing" and 0.0 <= s.value <= 1.0
    assert qc.from_routing(None) is None          # missing → neutral (None)
    assert qc.from_routing("bogus") is None        # unknown level → neutral


def test_from_resolution_only_when_references_present():
    class _Res:
        refs = []
        confidence = 0.0
        needs_clarification = False
    assert qc.from_resolution(_Res()) is None      # no refs → no signal

    class _Res2:
        refs = ["it"]
        confidence = 0.8
        needs_clarification = False
    sig = qc.from_resolution(_Res2())
    assert sig is not None and sig.source == "reference" and sig.value == 0.8


def test_from_trust_reuses_trust_shape():
    res = confidence_band(ConfidenceSignals(goal_passed=True,
                                            verify_attempted=True, verify_ok=True))
    sig = qc.from_trust(res)
    assert sig.source == "trust" and sig.value == res.score


# ── aggregation (Property 4) ─────────────────────────────────────────────────
def test_aggregate_no_signals_is_neutral_medium():
    r = qc.aggregate([])
    assert isinstance(r, ConfidenceResult)
    assert r.band == "medium"                      # neutral, never a failure


def test_aggregate_skips_none_signals():
    r = qc.aggregate([None, qc.from_routing("trivial"), None])
    assert r.band == "high"                        # trivial routing → high


def test_aggregate_blends_weighted():
    low_trust = qc.SubsystemConfidence("trust", 0.2, ["build failing"])
    high_route = qc.SubsystemConfidence("routing", 0.9, ["trivial"])
    r = qc.aggregate([low_trust, high_route])
    # trust is weighted higher, so the blend leans toward the low signal.
    assert r.score < 0.75


# ── gating (Property 5) ──────────────────────────────────────────────────────
def test_gate_maps_bands_to_control_flow():
    assert qc.gate(ConfidenceResult("high", 0.9)) == "proceed"
    assert qc.gate(ConfidenceResult("low", 0.2)) == "clarify"
    assert qc.gate(ConfidenceResult("medium", 0.6)) == "judgment"


def test_aggregate_failopen_on_bad_signal():
    class _Bad:
        source = "x"
        def clamped(self):
            raise RuntimeError("boom")
    # Must not raise — fail-open to neutral (Property 9).
    r = qc.aggregate([_Bad()])
    assert r.band in ("high", "medium", "low")
