"""Tests for panel diarization (vNext §4.16, Stage 7 Component K)."""
from __future__ import annotations

import app.live.panel as P


def _d(**kw):
    return P.PanelDiarizer(merge_threshold=0.7, max_speakers=4, **kw)


# ---- online clustering ----------------------------------------------------
def test_same_voice_stays_one_speaker():
    p = _d()
    a = p.assign([1.0, 0.0, 0.0], text="q1")
    b = p.assign([0.96, 0.04, 0.0], text="q2")   # cosine ~1 with a
    assert a.id == b.id == "P1"
    assert p.panel_size() == 1


def test_distinct_voice_spawns_new_speaker():
    p = _d()
    p.assign([1.0, 0.0, 0.0], text="q1")
    s = p.assign([0.0, 1.0, 0.0], text="q2")      # orthogonal → new
    assert s.id == "P2"
    assert p.is_panel() is True


def test_three_distinct_speakers():
    p = _d()
    ids = {
        p.assign([1.0, 0.0, 0.0]).id,
        p.assign([0.0, 1.0, 0.0]).id,
        p.assign([0.0, 0.0, 1.0]).id,
    }
    assert ids == {"P1", "P2", "P3"}
    assert p.panel_size() == 3


def test_returning_voice_reattaches_not_respawns():
    p = _d()
    p.assign([1.0, 0.0, 0.0])          # P1
    p.assign([0.0, 1.0, 0.0])          # P2
    back = p.assign([0.97, 0.03, 0.0])  # P1's voice returns
    assert back.id == "P1"
    assert p.panel_size() == 2          # did NOT create a 3rd


def test_max_speakers_cap_attaches_to_nearest():
    p = P.PanelDiarizer(merge_threshold=0.99, max_speakers=2)
    p.assign([1.0, 0.0, 0.0])          # P1
    p.assign([0.0, 1.0, 0.0])          # P2 (cap reached)
    s = p.assign([0.0, 0.0, 1.0])      # would be a 3rd, but capped
    assert s.id in ("P1", "P2")
    assert p.panel_size() == 2


# ---- fail-soft to single interviewer --------------------------------------
def test_no_embedding_is_single_p1():
    p = P.PanelDiarizer()
    s = p.assign(None, text="hello")
    assert s.id == "P1"
    assert p.panel_size() == 1
    assert not p.is_panel()


def test_zero_vector_fails_soft():
    p = P.PanelDiarizer()
    s = p.assign([0.0, 0.0, 0.0], text="x")
    assert s.id == "P1"


def test_garbage_embedding_never_raises():
    p = P.PanelDiarizer()
    s = p.assign(["not", "a", "number"], text="x")  # type: ignore[list-item]
    assert s.id == "P1"


# ---- per-speaker role / situation / turns ---------------------------------
def test_role_and_situation_attach_to_slot():
    p = _d()
    s = p.assign([1.0, 0.0, 0.0], text="q", role="hiring_manager",
                 situation="conviction_trap")
    assert s.role == "hiring_manager"
    assert s.situation == "conviction_trap"
    assert s.turns == ["q"]


def test_turns_ring_buffer_bounded():
    p = _d()
    for i in range(60):
        p.assign([1.0, 0.0, 0.0], text=f"t{i}")
    assert len(p.slots[0].turns) <= 50


def test_describe_shape():
    p = _d()
    p.assign([1.0, 0.0, 0.0], text="q", role="primary_interviewer")
    d = p.describe()
    assert d and set(d[0]) == {"id", "count", "role", "situation", "turns"}


# ---- tracker attachment ---------------------------------------------------
def test_for_tracker_is_stable():
    class T:
        pass
    t = T()
    a = P.for_tracker(t)
    b = P.for_tracker(t)
    assert a is b                       # one diarizer per session
