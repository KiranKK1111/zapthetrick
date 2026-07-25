"""Tests for the emotion engine (vNext §10.6.4, Stage 10 Component B)."""
from __future__ import annotations

import app.live.emotion_engine as E


def _on(monkeypatch):
    monkeypatch.setattr(E, "enabled", lambda: True)


# ---- SER read (injected head) --------------------------------------------
def test_ser_read_normalizes_and_picks_top():
    r = E.ser_read({}, ser_fn=lambda f: {"nervous": 6, "calm": 3, "confident": 1})
    assert r.top == "nervous"
    assert abs(sum(r.distribution.values()) - 1.0) < 1e-6   # normalized
    assert 0 < r.confidence <= 1


def test_ser_read_no_head_is_neutral():
    assert E.ser_read({"pitch": 1}).top == E.NEUTRAL


def test_ser_read_arousal_and_valence():
    r = E.ser_read({}, ser_fn=lambda f: {"excited": 0.8, "calm": 0.2})
    assert r.arousal > 0.5 and r.valence > 0     # excited = high arousal, +valence
    r2 = E.ser_read({}, ser_fn=lambda f: {"sad": 0.9, "calm": 0.1})
    assert r2.valence < 0                        # sad = negative valence


def test_ser_read_filters_unknown_labels():
    r = E.ser_read({}, ser_fn=lambda f: {"bogus": 0.9, "calm": 0.1})
    assert r.top == "calm"                       # bogus dropped


def test_ser_read_never_raises():
    assert E.ser_read({}, ser_fn=lambda f: 1 / 0).top == E.NEUTRAL


# ---- fusion ---------------------------------------------------------------
def test_fusion_agreement_reinforces():
    r = E.ser_read({}, ser_fn=lambda f: {"nervous": 0.6, "calm": 0.4})
    fused = E.fuse(r, transcript_sentiment=-0.6)   # both negative → agree
    assert fused.agree and fused.confidence >= r.confidence


def test_fusion_disagreement_widens_band():
    r = E.ser_read({}, ser_fn=lambda f: {"nervous": 0.6, "calm": 0.4})
    fused = E.fuse(r, transcript_sentiment=+0.7)   # nervous voice, positive words
    assert not fused.agree
    assert fused.confidence < r.confidence         # band widened


def test_fusion_situation_prior_nudges_nervous():
    r = E.ser_read({}, ser_fn=lambda f: {"nervous": 0.5, "calm": 0.5})
    base = E.fuse(r).confidence
    nudged = E.fuse(r, situation="conviction_trap").confidence
    assert nudged >= base


def test_fusion_never_raises():
    assert E.fuse(E.EmotionRead()).label == E.NEUTRAL


# ---- hysteretic session state --------------------------------------------
def test_session_needs_persistence_to_flip():
    st = E.SessionEmotionState(persistence=3, min_conf=0.5)
    st.update(E.FusedEmotion("nervous", 0.6))
    st.update(E.FusedEmotion("nervous", 0.6))
    assert st.label == E.NEUTRAL                  # only 2 reads
    st.update(E.FusedEmotion("nervous", 0.6))
    assert st.label == "nervous"                  # 3rd → flip


def test_one_shaky_read_does_not_flip():
    st = E.SessionEmotionState(persistence=3, min_conf=0.5)
    st.update(E.FusedEmotion("nervous", 0.6))
    st.update(E.FusedEmotion("calm", 0.6))
    st.update(E.FusedEmotion("nervous", 0.6))
    assert st.label == E.NEUTRAL                  # not 3 consecutive


def test_low_confidence_reads_dont_persist():
    st = E.SessionEmotionState(persistence=2, min_conf=0.6)
    st.update(E.FusedEmotion("nervous", 0.3))
    st.update(E.FusedEmotion("nervous", 0.3))
    assert st.label == E.NEUTRAL                  # below min_conf → treated neutral


def test_dismissal_overrides_for_session():
    st = E.SessionEmotionState(persistence=1, min_conf=0.5)
    st.update(E.FusedEmotion("nervous", 0.9))
    assert st.label == "nervous"
    st.override()
    assert st.overridden and st.label == E.NEUTRAL
    assert st.update(E.FusedEmotion("nervous", 0.95)) == E.NEUTRAL   # stays dropped


def test_is_dismissal_cues():
    for t in ("I'm fine", "no I am fine thanks", "don't worry, all good"):
        assert E.is_dismissal(t)
    assert not E.is_dismissal("I'm really struggling here")


def test_is_dismissal_semantic_injected():
    assert E.is_dismissal("anything", classify_fn=lambda t: True)
    assert not E.is_dismissal("I'm fine", classify_fn=lambda t: False)


# ---- response policy ------------------------------------------------------
def _nervous_state(conf=0.8):
    st = E.SessionEmotionState()
    st.label = "nervous"
    st.confidence = conf
    return st


def test_policy_acknowledges_when_strong_agree_and_surface(monkeypatch):
    _on(monkeypatch)
    p = E.response_policy(_nervous_state(0.8), surface="live", agree=True)
    assert p.acknowledge and "take your time" in p.acknowledgement.lower()
    assert p.register


def test_policy_silent_on_wrong_surface(monkeypatch):
    _on(monkeypatch)
    p = E.response_policy(_nervous_state(0.8), surface="chat", agree=True)
    assert not p.acknowledge                     # register shifts silently
    assert p.register


def test_policy_silent_when_disagree(monkeypatch):
    _on(monkeypatch)
    p = E.response_policy(_nervous_state(0.8), surface="live", agree=False)
    assert not p.acknowledge


def test_policy_silent_when_weak_confidence(monkeypatch):
    _on(monkeypatch)
    p = E.response_policy(_nervous_state(0.6), surface="live", agree=True)
    assert not p.acknowledge                     # below ack_min_conf


def test_policy_neutral_when_overridden(monkeypatch):
    _on(monkeypatch)
    st = _nervous_state(0.9)
    st.override()
    p = E.response_policy(st, surface="live", agree=True)
    assert not p.acknowledge and p.register == ""


def test_policy_disabled_is_silent(monkeypatch):
    monkeypatch.setattr(E, "enabled", lambda: False)
    p = E.response_policy(_nervous_state(0.9), surface="live", agree=True)
    assert p.register == "" and not p.acknowledge


def test_policy_never_clinical(monkeypatch):
    _on(monkeypatch)
    p = E.response_policy(_nervous_state(0.8), surface="live", agree=True)
    # A tentative supportive line, never a diagnosis.
    assert "you sound" not in p.acknowledgement.lower()
    assert "anxious" not in p.acknowledgement.lower()


def test_policy_never_raises(monkeypatch):
    _on(monkeypatch)
    assert isinstance(E.response_policy(E.SessionEmotionState()), E.EmotionPolicy)
