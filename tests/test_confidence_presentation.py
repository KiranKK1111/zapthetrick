"""Confidence-based UX presentation variation (roadmap Phase 5 #20).

Pins that the presentation gets progressively more transparent as confidence
drops: high → clean/brief, medium → hedge+badge, low → hedge+badge+alternatives+
regenerate+detail. Fail-open.
"""
from __future__ import annotations

from app.chat.trust import ConfidenceResult
from app.quality.confidence import Presentation, presentation


def _res(band, score):
    return ConfidenceResult(band=band, score=score, reasons=[])


def test_high_confidence_is_clean_and_brief():
    p = presentation(_res("high", 0.9))
    assert isinstance(p, Presentation)
    assert p.hedge == "" and not p.show_confidence
    assert not p.offer_alternatives and not p.offer_regenerate
    assert p.verbosity == "brief"


def test_medium_confidence_hedges_and_badges():
    p = presentation(_res("medium", 0.6))
    assert p.hedge and p.show_confidence and p.offer_alternatives
    assert not p.offer_regenerate
    assert p.verbosity == "normal"


def test_low_confidence_offers_regenerate_and_detail():
    p = presentation(_res("low", 0.3))
    assert p.hedge and p.show_confidence
    assert p.offer_alternatives and p.offer_regenerate
    assert p.verbosity == "detailed"


def test_as_dict_is_serialisable():
    d = presentation(_res("low", 0.2)).as_dict()
    assert set(d) >= {"band", "score", "hedge", "show_confidence",
                      "offer_alternatives", "offer_regenerate", "verbosity"}


def test_failopen_on_bad_result():
    class _Bad:
        band = "high"
        score = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
    p = presentation(_Bad())
    assert isinstance(p, Presentation)
