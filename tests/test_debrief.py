"""Tests for the assisted-session debrief (vNext §4.17, Stage 11 Component D)."""
from __future__ import annotations

import app.live.debrief as D


def _on(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)


_DELIVERIES = [
    {"question": "Explain Kafka", "delivered_ratio": 1.0, "completed": True, "improvised": []},
    {"question": "Design a limiter", "delivered_ratio": 0.5, "completed": False,
     "improvised": ["I scaled it in prod"]},
    {"question": "What is your expected salary", "delivered_ratio": 1.0},
]
_CLAIMS = ["Kafka is a log", "Kafka is a log", "My CTC was 30 LPA", "I used B-trees"]
_SITUATIONS = [{"situation": "stress", "confidence": 0.8},
               {"situation": "salary", "confidence": 0.9}, "rapport"]
_PANEL = [{"id": "P1", "role": "primary_interviewer", "turns": 5},
          {"id": "P2", "role": "", "turns": 2}]
_TOPICS = ["Kafka", "Postgres", "salary negotiation"]


def _build(monkeypatch, **kw):
    _on(monkeypatch)
    return D.build_debrief(deliveries=_DELIVERIES, claims=_CLAIMS,
                           situations=_SITUATIONS, panel=_PANEL, topics=_TOPICS, **kw)


# ---- descriptive + private invariants ------------------------------------
def test_debrief_is_descriptive_not_scored(monkeypatch):
    d = _build(monkeypatch)
    assert d.scored is False
    assert "not a score" in d.to_dict()["disclaimer"].lower()


def test_debrief_private_by_default(monkeypatch):
    assert _build(monkeypatch).private is True


def test_disabled_is_empty_but_private(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: False)
    d = D.build_debrief(deliveries=_DELIVERIES, claims=_CLAIMS)
    assert d.delivery_map == [] and d.claims == [] and d.private is True


# ---- comp exclusion (§11.3) ----------------------------------------------
def test_comp_excluded_from_delivery(monkeypatch):
    d = _build(monkeypatch)
    qs = [x["question"] for x in d.delivery_map]
    assert "Explain Kafka" in qs
    assert not any("salary" in q.lower() for q in qs)


def test_comp_excluded_from_claims(monkeypatch):
    d = _build(monkeypatch)
    assert "I used B-trees" in d.claims
    assert not any("ctc" in c.lower() or "lpa" in c.lower() for c in d.claims)


def test_salary_situation_excluded(monkeypatch):
    d = _build(monkeypatch)
    names = [s["situation"] for s in d.situations]
    assert "stress" in names and "salary" not in names


def test_comp_topic_excluded_from_followups(monkeypatch):
    d = _build(monkeypatch)
    assert not any("salary" in f.lower() for f in d.follow_ups)


# ---- section assembly -----------------------------------------------------
def test_delivery_map_ratio_and_interruption(monkeypatch):
    d = _build(monkeypatch)
    lim = next(x for x in d.delivery_map if "limiter" in x["question"])
    assert lim["delivered_ratio"] == 0.5 and lim["completed"] is False
    assert lim["improvised"] == 1


def test_claims_deduped(monkeypatch):
    d = _build(monkeypatch)
    assert d.claims.count("Kafka is a log") == 1


def test_situation_replay_preserves_strings(monkeypatch):
    d = _build(monkeypatch)
    assert {"situation": "rapport"} in d.situations


def test_panel_dynamics(monkeypatch):
    d = _build(monkeypatch)
    assert d.panel[0]["id"] == "P1" and d.panel[0]["turns"] == 5


def test_followups_from_topics(monkeypatch):
    d = _build(monkeypatch)
    assert any("Kafka" in f for f in d.follow_ups)


def test_followups_use_injected_predictor(monkeypatch):
    _on(monkeypatch)
    d = D.build_debrief(topics=["Kafka"],
                        predict_fn=lambda ts: ["What breaks Kafka at 1M msg/s?"])
    assert d.follow_ups == ["What breaks Kafka at 1M msg/s?"]


def test_build_never_raises(monkeypatch):
    _on(monkeypatch)
    assert isinstance(D.build_debrief(deliveries=[None], claims=[None]),
                      D.SessionDebrief)


# ---- markdown render ------------------------------------------------------
def test_markdown_has_sections_and_disclaimer(monkeypatch):
    md = D.to_markdown(_build(monkeypatch))
    assert md.startswith("# Session debrief")
    assert "not a score" in md.lower()
    assert "## Delivery" in md and "## What you said" in md
    assert "## Practice these follow-ups" in md


def test_markdown_shows_interruption(monkeypatch):
    md = D.to_markdown(_build(monkeypatch))
    assert "interrupted" in md and "improvised" in md


def test_markdown_empty_debrief():
    md = D.to_markdown(D.SessionDebrief())
    assert md.startswith("# Session debrief")
