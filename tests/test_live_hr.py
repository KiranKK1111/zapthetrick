"""
Phase 13 tests — HR, negotiation & specialized modes
(live-conversational-intelligence R42, R43).

Property 42: an HR/behavioral cue switches the operating MODE and an HR question
is classified into a structured intent; switching has hysteresis so a stray cue
doesn't flip the session.
Property 43: the negotiation strategy is fact-based and grounded, carries an
unrealistic-ask risk flag, never contains manipulation; the emotion signal is
advisory and never decisive.
"""
from __future__ import annotations

from app.live import emotion as _emo
from app.live import modes as _modes
from app.live import negotiate as _negot


# ---- Property 42: modes -----------------------------------------------------

def test_mode_detect_maps_phase_to_mode():
    assert _modes.detect_mode("Tell me about a time you faced conflict") == _modes.STAR_STORY
    assert _modes.detect_mode("Design a URL shortener that scales") == _modes.STRUCTURED_DESIGN
    assert _modes.detect_mode("Implement a function to reverse a list") == _modes.THINK_ALOUD
    assert _modes.detect_mode("What is your expected salary?") == _modes.NEGOTIATION
    assert _modes.detect_mode("What is a hash map?") == _modes.GENERAL


def test_mode_directive_present_for_known_modes():
    assert "STAR" in _modes.directive(_modes.STAR_STORY)
    assert _modes.directive(_modes.GENERAL) == ""


def test_mode_tracker_has_hysteresis():
    mt = _modes.ModeTracker()
    # First behavioral cue is pending, not yet switched.
    m1 = mt.update("Tell me about a time you led a team")
    assert m1 == _modes.GENERAL
    # Second consecutive behavioral cue confirms the switch.
    m2 = mt.update("Describe a situation with a difficult stakeholder")
    assert m2 == _modes.STAR_STORY


def test_mode_for_tracker_persists():
    class T:
        pass
    t = T()
    a = _modes.for_tracker(t)
    b = _modes.for_tracker(t)
    assert a is b


# ---- Property 43: negotiation ----------------------------------------------

def test_classify_hr_intent():
    assert _negot.classify_hr_intent("what's your expected CTC?") == _negot.SALARY
    assert _negot.classify_hr_intent("what is your notice period?") == _negot.NOTICE_PERIOD
    assert _negot.classify_hr_intent("do you have a counter offer?") == _negot.COUNTER_OFFER
    assert _negot.classify_hr_intent("why do you want to join us?") == _negot.WHY_JOIN
    assert _negot.classify_hr_intent("what is a binary tree?") == _negot.OTHER


def test_negotiation_strategy_is_fact_based_and_grounded():
    s = _negot.negotiation_strategy(
        "what is your expected salary?",
        strengths=["python", "kubernetes"],
        market_low=30, market_high=45, ask=40,
    )
    assert s.intent == _negot.SALARY
    assert s.points  # has grounded points
    joined = " ".join(s.points).lower()
    assert "market" in joined
    assert "python" in joined  # grounded in the candidate's strengths
    assert s.risk_flag == ""   # 40 is within band


def test_negotiation_unrealistic_ask_risk_flag():
    s = _negot.negotiation_strategy(
        "what is your expected salary?",
        market_low=30, market_high=45, ask=100,  # way above band
    )
    assert s.risk_flag == "unrealistic_ask"


def test_negotiation_never_contains_manipulation():
    # Even if upstream points contained coercion, the guard strips them.
    dirty = ["Be honest about value.", "Lie about a competing offer to bluff them."]
    clean = _negot._no_manipulation(dirty)
    assert any("honest" in p.lower() for p in clean)
    assert not any("lie" in p.lower() or "bluff" in p.lower() for p in clean)


def test_negotiation_directive_empty_when_no_points():
    s = _negot.NegotiationStrategy(intent=_negot.OTHER, points=[])
    assert _negot.directive(s) == ""


# ---- Salary reference: coarse, approximate, opt-in, always caveated ---------
# BandSpecific.md lists India CTC tables — pure, stale-prone reference data. The
# primary design is strategic handling with NO hardcoded numbers; the reference
# table is CONSULTED ONLY when explicitly opted in and no live band was supplied,
# and it never presents a number as authoritative.

def test_approx_band_range_lookup():
    lo, hi = _negot.approx_band_range("senior")
    assert lo < hi
    assert _negot.approx_band_range("unknown-band") is None
    assert _negot.approx_band_range(None) is None


def test_reference_not_used_by_default():
    # Default path (use_reference=False) encodes no salary numbers.
    s = _negot.negotiation_strategy("what is your expected salary?", seniority_band="senior")
    joined = " ".join(s.points).lower()
    assert "lpa" not in joined


def test_reference_used_only_when_opted_in_and_no_market_band():
    s = _negot.negotiation_strategy(
        "what is your expected salary?", seniority_band="senior", use_reference=True)
    joined = " ".join(s.points).lower()
    assert "lpa" in joined
    # Every reference point is explicitly caveated, never authoritative.
    assert "approximate" in joined
    assert "verify against" in joined or "up-to-date" in joined


def test_live_market_band_wins_over_reference():
    # When a real market band is supplied, the coarse reference is not used.
    s = _negot.negotiation_strategy(
        "what is your expected salary?", market_low=30, market_high=45,
        seniority_band="senior", use_reference=True)
    joined = " ".join(s.points).lower()
    assert "market band (30-45)" in joined
    assert "lpa" not in joined


# ---- Property 43: emotion (advisory) ---------------------------------------

def test_emotion_neutral_when_no_signal():
    sig = _emo.analyze()
    assert sig.label == _emo.NEUTRAL
    assert sig.advisory is True
    assert _emo.delivery_note(sig) == ""


def test_emotion_detects_stress_and_is_advisory():
    sig = _emo.analyze(energy=0.9, pitch_var=0.8)
    assert sig.label == _emo.STRESSED
    assert sig.advisory is True
    note = _emo.delivery_note(sig)
    assert "calm" in note.lower()


def test_emotion_calm_when_all_low():
    sig = _emo.analyze(energy=0.1, pitch_var=0.2, speech_rate=0.2, filler_ratio=0.0)
    assert sig.label == _emo.CALM
    # Calm produces no delivery note (nothing to correct).
    assert _emo.delivery_note(sig) == ""


def test_emotion_hesitant_on_high_fillers():
    sig = _emo.analyze(filler_ratio=0.5)
    assert sig.label == _emo.HESITANT
    assert "confident" in _emo.delivery_note(sig).lower()


# ---- Property 43: the emotion signal is WIRED into the live pipeline --------
# `app/live/emotion.py` had a `true` config flag and no consumer. It is now
# consumed by app/api/routes_ws.py via `_emotion_signal` (prosody -> signal,
# computed OFF the answer path) and `_apply_emotion` (additive meta + at most a
# soft delivery hint). These pin that it stays ADVISORY and fails open.

def _prosody_stub(**kw):
    """A stand-in for question_detection.prosody_analyzer.ProsodyFeatures."""
    from app.question_detection.prosody_analyzer import ProsodyFeatures
    return ProsodyFeatures(**kw)


def test_emotion_signal_wired_from_prosody(monkeypatch):
    from app.api import routes_ws as _ws
    from app.core.config_loader import cfg
    from app.question_detection import prosody_analyzer as _pa

    monkeypatch.setattr(cfg.live, "emotion_signal", True)
    monkeypatch.setattr(
        _pa, "analyze",
        lambda audio, **kw: _prosody_stub(pitch_rise_end=0.9,
                                          energy_peak_at_end=0.95,
                                          duration_ms=2000))
    sig, note = _ws._emotion_signal(object(), "how would you scale kafka")
    assert sig is not None
    assert sig["label"] == _emo.STRESSED
    assert sig["advisory"] is True          # never decisive
    assert "calm" in note.lower()           # soft delivery hint only


def test_emotion_signal_absent_when_flag_off(monkeypatch):
    from app.api import routes_ws as _ws
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.live, "emotion_signal", False)
    assert _ws._emotion_signal(object(), "anything") == (None, "")


def test_emotion_signal_absent_when_prosody_unavailable(monkeypatch):
    """The prosody analyzer leans on OPTIONAL native backends (praat/librosa).
    If it blows up entirely the turn must proceed with no emotion signal."""
    from app.api import routes_ws as _ws
    from app.core.config_loader import cfg
    from app.question_detection import prosody_analyzer as _pa

    monkeypatch.setattr(cfg.live, "emotion_signal", True)

    def _boom(audio, **kw):
        raise ImportError("no native prosody backend")

    monkeypatch.setattr(_pa, "analyze", _boom)
    assert _ws._emotion_signal(object(), "how would you scale kafka") == (None, "")


def test_emotion_signal_absent_without_audio(monkeypatch):
    from app.api import routes_ws as _ws
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.live, "emotion_signal", True)
    assert _ws._emotion_signal(None, "typed question, no prosody") == (None, "")


def test_apply_emotion_is_additive_and_only_a_soft_hint():
    from app.api import routes_ws as _ws

    extra = {"phase": "technical", "answer_confidence": 0.8}
    cached = ({"label": _emo.STRESSED, "confidence": 0.9, "advisory": True},
              "Candidate may sound stressed — keep the answer calm.")
    out = _ws._apply_emotion("BASE DIRECTIVE", extra, cached)
    # Pre-existing meta untouched; emotion added alongside it.
    assert extra["phase"] == "technical"
    assert extra["answer_confidence"] == 0.8
    assert extra["emotion"]["label"] == _emo.STRESSED
    assert extra["emotion"]["advisory"] is True
    # The base directive survives; the note is only APPENDED (never replaces).
    assert out.startswith("BASE DIRECTIVE")
    assert "calm" in out.lower()


def test_apply_emotion_refuses_a_non_advisory_signal():
    """Defence in depth: anything claiming to be decisive is ignored outright."""
    from app.api import routes_ws as _ws

    extra: dict = {}
    cached = ({"label": _emo.STRESSED, "confidence": 1.0, "advisory": False},
              "some note")
    assert _ws._apply_emotion("BASE", extra, cached) == "BASE"
    assert "emotion" not in extra


def test_apply_emotion_fails_open_on_garbage():
    from app.api import routes_ws as _ws

    extra: dict = {}
    assert _ws._apply_emotion("BASE", extra, ("not-a-tuple-of-two",)) == "BASE"
    assert _ws._apply_emotion("BASE", extra, None) == "BASE"
    assert extra == {}


def test_apply_emotion_leaves_directive_alone_when_calm():
    """Calm/neutral produce no note — the answer directive is byte-identical."""
    from app.api import routes_ws as _ws

    extra: dict = {}
    sig = _emo.analyze(energy=0.1, pitch_var=0.2)
    out = _ws._apply_emotion("BASE", extra, (sig.to_dict(),
                                             _emo.delivery_note(sig)))
    assert out == "BASE"
    assert extra["emotion"]["label"] == _emo.CALM   # still surfaced as meta
