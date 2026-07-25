"""Tests for situational intelligence + two-lane display (vNext §4.15, Stage 7 J)."""
from __future__ import annotations

import app.live.situational as S


# ---- situation classifier -------------------------------------------------
def test_semantic_classify_wins():
    r = S.classify_situation("are you totally certain about that",
                             classify_fn=lambda q: S.CONVICTION_TRAP)
    assert r.situation == S.CONVICTION_TRAP
    assert r.source == "semantic"


def test_semantic_ignores_out_of_taxonomy_class():
    # A classifier that returns a non-situation string → falls through to cues.
    r = S.classify_situation("what's your salary expectation",
                             classify_fn=lambda q: "banana")
    assert r.situation == S.SALARY
    assert r.source == "fallback"


def test_fallback_cues_when_semantic_unavailable():
    r = S.classify_situation("we're short on time, be fast please",
                             classify_fn=lambda q: None)
    assert r.situation == S.STRESS
    assert r.source == "fallback"


def test_emotion_only_nudges_to_stress():
    r = S.classify_situation("", emotion_label="stressed", classify_fn=lambda q: None)
    assert r.situation == S.STRESS
    assert r.source == "emotion"


def test_neutral_when_nothing_matches():
    r = S.classify_situation("could you describe a red-black tree",
                             classify_fn=lambda q: None)
    assert r.situation == S.NEUTRAL
    assert r.confidence == 0.0


def test_classify_never_raises():
    r = S.classify_situation(None, classify_fn=lambda q: 1 / 0)  # type: ignore[arg-type]
    assert r.situation == S.NEUTRAL


# ---- per-situation × band strategy ----------------------------------------
def test_strategy_has_directive_and_chips():
    st = S.strategy_for(S.SituationRead(S.HARSHNESS, 0.8), band="mid")
    assert st.directive
    assert st.chips and all(isinstance(c, S.GuidanceChip) for c in st.chips)


def test_band_shades_conviction_directive():
    jr = S.strategy_for(S.SituationRead(S.CONVICTION_TRAP), band="intern").directive
    sr = S.strategy_for(S.SituationRead(S.CONVICTION_TRAP), band="principal").directive
    assert jr != sr                      # band actually changes the posture


def test_strategy_accepts_bare_situation_string():
    st = S.strategy_for(S.SALARY, band="senior")
    assert st.situation == S.SALARY
    assert st.directive


def test_neutral_strategy_is_empty():
    st = S.strategy_for(S.SituationRead(S.NEUTRAL), band="mid")
    assert st.directive == ""
    assert st.chips == []


# ---- two-lane display contract + validator --------------------------------
def test_guidance_chip_is_never_spoken():
    c = S.GuidanceChip("hold your ground", spoken=True)  # even if constructed True…
    d = S.build_display("some answer", [c])
    assert all(not ch.spoken for ch in d.guidance)       # …build coerces it False
    assert d.to_dict()["guidance"][0]["spoken"] is False


def test_validate_clean_separation_passes():
    d = S.TwoLaneDisplay("Kafka uses partitions.",
                         [S.GuidanceChip("stay calm, defend with evidence")])
    ok, violations = S.validate_separation(d)
    assert ok and violations == []


def test_validator_catches_guidance_leak_into_dictatable():
    d = S.TwoLaneDisplay("hold your ground here and stay firm",
                         [S.GuidanceChip("hold your ground here")])
    ok, violations = S.validate_separation(d)
    assert not ok
    assert any("leaked" in v for v in violations)


def test_validator_catches_spoken_flag_violation():
    d = S.TwoLaneDisplay("answer text", [S.GuidanceChip("coach", spoken=True)])
    ok, violations = S.validate_separation(d)
    assert not ok
    assert any("spoken" in v for v in violations)


def test_build_display_drops_leaked_and_dedups():
    d = S.build_display(
        "hold your ground",
        [S.GuidanceChip("hold your ground"),   # leaked → dropped
         S.GuidanceChip("slow down"),
         S.GuidanceChip("slow down")])          # dup → dropped
    texts = [c.text for c in d.guidance]
    assert texts == ["slow down"]


def test_build_display_result_validates_clean():
    st = S.strategy_for(S.SituationRead(S.STRESS), band="mid")
    d = S.build_display("Lead with the headline answer.", st.chips)
    ok, _ = S.validate_separation(d)
    assert ok


def test_validate_separation_fail_open_on_garbage():
    class Bad:
        dictatable = "x"
        guidance = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
    ok, violations = S.validate_separation(Bad())   # type: ignore[arg-type]
    assert ok and violations == []
