"""Tests for the design system engine (vNext §3.11, Stage 8 Component D)."""
from __future__ import annotations

import asyncio

import app.documents.design_system as DS


def _run(coro):
    return asyncio.run(coro)


# ---- WCAG maths -----------------------------------------------------------
def test_contrast_extremes():
    assert round(DS.contrast_ratio("#000000", "#ffffff"), 1) == 21.0
    assert round(DS.contrast_ratio("#ffffff", "#ffffff"), 1) == 1.0


def test_contrast_is_symmetric():
    assert DS.contrast_ratio("#123456", "#abcdef") == DS.contrast_ratio("#abcdef", "#123456")


def test_short_hex_supported():
    assert DS.contrast_ratio("#000", "#fff") == DS.contrast_ratio("#000000", "#ffffff")


def test_passes_wcag_thresholds():
    assert DS.passes_wcag("#000", "#fff")                      # 21:1
    assert not DS.passes_wcag("#999999", "#ffffff")            # 2.85:1 fails AA
    assert DS.passes_wcag("#999999", "#ffffff", large=True) is False   # 2.85 < 3.0
    assert DS.passes_wcag("#767676", "#ffffff")                # ~4.54 passes AA


def test_bad_colour_reads_as_fail():
    assert DS.contrast_ratio("not-a-color", "#fff") == 1.0


# ---- validation -----------------------------------------------------------
def test_default_theme_is_aa_safe():
    v = DS.validate_theme(DS.SAFE_DEFAULT)
    assert v.ok and v.violations == []


def test_bad_theme_flags_every_failing_pair():
    bad = DS.ThemeTokens(text="#bbbbbb", muted="#cccccc", accent="#88bbff",
                         on_accent="#dddddd")
    v = DS.validate_theme(bad)
    assert not v.ok
    assert len(v.violations) >= 3
    assert all("ratio" in vio for vio in v.violations)


def test_aaa_level_is_stricter():
    # ~4.54 on white: passes AA (4.5), fails AAA (7.0). surface=bg=white so both
    # text pairs are the same 4.54, isolating the level threshold.
    t = DS.ThemeTokens(text="#767676", surface="#ffffff")
    assert DS.validate_theme(t, level="AA").ok
    assert not DS.validate_theme(t, level="AAA").ok


# ---- auto-correct guarantees accessibility -------------------------------
def test_auto_correct_makes_bad_theme_safe():
    bad = DS.ThemeTokens(text="#bbbbbb", muted="#cccccc", accent="#88bbff",
                         on_accent="#dddddd", surface="#f0f0f0")
    fixed, corrections = DS.auto_correct(bad)
    assert DS.validate_theme(fixed).ok           # GUARANTEED safe
    assert corrections                            # something was changed


def test_auto_correct_reaches_fixpoint_on_shared_role():
    # `text` sits on both bg and surface; a single pass could leave one failing.
    t = DS.ThemeTokens(text="#a0a0a0", bg="#ffffff", surface="#eeeeee")
    fixed, _ = DS.auto_correct(t)
    v = DS.validate_theme(fixed)
    assert v.ok
    assert v.ratios["text/bg"] >= 4.5 and v.ratios["text/surface"] >= 4.5


def test_auto_correct_impossible_theme_still_safe():
    worse = DS.ThemeTokens(bg="#ffffff", surface="#ffffff", text="#fefefe",
                           muted="#fdfdfd", accent="#ffffff", on_accent="#ffffff")
    fixed, _ = DS.auto_correct(worse)
    assert DS.validate_theme(fixed).ok


def test_auto_correct_leaves_good_theme_untouched():
    fixed, corrections = DS.auto_correct(DS.SAFE_DEFAULT)
    assert corrections == []
    assert fixed.to_dict() == DS.SAFE_DEFAULT.to_dict()


# ---- font pairings + layout ----------------------------------------------
def test_font_pairing_known_and_default():
    assert DS.font_pairing("technical") == ("Space Grotesk", "IBM Plex Sans")
    assert DS.font_pairing("unknown") == DS.FONT_PAIRINGS["inter"]


def test_layout_variants_present():
    assert "comfortable" in DS.LAYOUT_VARIANTS and "compact" in DS.LAYOUT_VARIANTS


# ---- propose (injected) ---------------------------------------------------
class _Res:
    def __init__(self, obj):
        self.obj = obj


def test_propose_disabled_returns_safe_default(monkeypatch):
    monkeypatch.setattr(DS, "enabled", lambda: False)

    async def fn(schema, msgs, **kw):
        return _Res({"text": "#aaaaaa"})
    t, corr = _run(DS.propose_theme("navy report", propose_fn=fn))
    assert t.to_dict() == DS.SAFE_DEFAULT.to_dict()
    assert corr == []


def test_propose_enabled_is_always_accessible(monkeypatch):
    monkeypatch.setattr(DS, "enabled", lambda: True)

    async def fn(schema, msgs, **kw):
        return _Res({"bg": "#ffffff", "surface": "#f0f0f0", "text": "#aaaaaa",
                     "muted": "#bbbbbb", "accent": "#66aaff", "on_accent": "#eeeeee"})
    t, corr = _run(DS.propose_theme("x", propose_fn=fn))
    assert DS.validate_theme(t).ok           # a bad proposal is corrected to safe
    assert corr


def test_propose_fail_open_on_raise(monkeypatch):
    monkeypatch.setattr(DS, "enabled", lambda: True)

    async def boom(schema, msgs, **kw):
        raise RuntimeError("llm down")
    t, corr = _run(DS.propose_theme("x", propose_fn=boom))
    assert t.to_dict() == DS.SAFE_DEFAULT.to_dict()


def test_propose_fail_open_on_bad_obj(monkeypatch):
    monkeypatch.setattr(DS, "enabled", lambda: True)

    async def fn(schema, msgs, **kw):
        return _Res("not a dict")
    t, _ = _run(DS.propose_theme("x", propose_fn=fn))
    assert t.to_dict() == DS.SAFE_DEFAULT.to_dict()


# ---- design score ---------------------------------------------------------
def test_design_score_rewards_safe_theme():
    s = DS.design_score(DS.SAFE_DEFAULT)
    assert s["wcag_ok"] and s["contrast"] == 100 and s["score"] >= 90


def test_design_score_penalizes_bad_contrast():
    bad = DS.ThemeTokens(text="#cccccc", muted="#dddddd")
    s = DS.design_score(bad)
    assert not s["wcag_ok"]
    assert s["contrast"] < 100


def test_design_score_never_raises():
    s = DS.design_score(DS.ThemeTokens(text="bad", bg="also-bad"))
    assert "score" in s
