"""Tests for the Phase-4 reasoning layers (advanced-intent-reasoning):
multi-interpretation, expected-reply simulation, latent prediction, critic.
Pure functions — no DB, no LLM."""
from __future__ import annotations

from app.clarify.critic import review
from app.clarify.interpretations import parse_interpretations, pick_interpretation
from app.clarify.latent import suggest
from app.clarify.simulation import questions_to_assumptions, to_assumption


def _q(header, *labels, recommended_idx=None, blocking=False):
    opts = [{"id": f"o{i}", "label": l, "description": "",
             "recommended": (i == recommended_idx)}
            for i, l in enumerate(labels)]
    return {"id": header.lower(), "header": header, "question": f"{header}?",
            "options": opts, "blocking": blocking}


class TestInterpretations:
    def test_parse_normalises_probabilities(self):
        out = parse_interpretations([
            {"reading": "A", "probability": 2},
            {"reading": "B", "probability": 2},
        ])
        assert abs(sum(i["probability"] for i in out) - 1.0) < 1e-9
        assert out[0]["probability"] == 0.5

    def test_single_interpretation_is_dropped(self):
        assert parse_interpretations([{"reading": "only"}]) == []

    def test_malformed_returns_empty(self):
        assert parse_interpretations("nope") == []
        assert parse_interpretations([{"x": 1}]) == []

    def test_dominant_reading_answers(self):
        kind, payload = pick_interpretation([
            {"reading": "git reset", "probability": 0.8},
            {"reading": "factory reset", "probability": 0.2},
        ])
        assert kind == "answer"
        assert payload == "git reset"

    def test_no_dominant_asks_with_options(self):
        kind, payload = pick_interpretation([
            {"reading": "git reset", "probability": 0.5},
            {"reading": "factory reset", "probability": 0.5},
        ])
        assert kind == "ask"
        assert [o["label"] for o in payload] == ["git reset", "factory reset"]
        assert sum(o["recommended"] for o in payload) <= 1

    def test_empty_is_none(self):
        assert pick_interpretation([]) == ("none", None)


class TestSimulation:
    def test_converts_with_recommended(self):
        a = to_assumption(_q("Language", "Python", "Java", recommended_idx=0))
        assert a == {"id": "language", "label": "Language", "value": "Python"}

    def test_no_recommended_no_assumption(self):
        assert to_assumption(_q("Language", "Python", "Java")) is None

    def test_blocking_never_converted(self):
        q = _q("Confirm", "Yes", "No", recommended_idx=0, blocking=True)
        assert to_assumption(q) is None

    def test_bulk(self):
        qs = [_q("Language", "Python", "Java", recommended_idx=0),
              _q("DB", "Postgres", "Mongo")]  # second has no recommendation
        out = questions_to_assumptions(qs)
        assert len(out) == 1
        assert out[0]["value"] == "Python"


class TestCritic:
    def test_drops_known_slot_question(self):
        qs = [_q("Language", "Python", "Java", recommended_idx=0),
              _q("Platform", "Web", "Mobile", recommended_idx=0)]
        out = review(qs, suppressed=["language"], known={})
        assert [q["header"] for q in out] == ["Platform"]

    def test_known_dict_also_suppresses(self):
        qs = [_q("Framework", "React", "Vue", recommended_idx=0)]
        out = review(qs, suppressed=[], known={"framework": "react"})
        assert out == []

    def test_drops_low_option_question(self):
        thin = {"id": "x", "header": "X", "question": "X?", "options": [
            {"id": "o0", "label": "only"}]}
        assert review([thin], suppressed=[]) == []

    def test_keeps_blocking(self):
        q = _q("Confirm", "Yes", "No", recommended_idx=0, blocking=True)
        # Even if its slot were suppressed, a blocking card is kept (R9.4).
        assert review([q], suppressed=["confirm"]) == [q]

    def test_deterministic(self):
        qs = [_q("Language", "Python", "Java", recommended_idx=0),
              _q("Platform", "Web", "Mobile", recommended_idx=0)]
        a = review(qs, suppressed=["language"])
        b = review(qs, suppressed=["language"])
        assert a == b


class TestLatent:
    def test_code_gen_suggestions(self):
        s = suggest("code_generation", {})
        assert s and len(s) <= 3
        assert any("test" in x.lower() for x in s)

    def test_chitchat_has_none(self):
        assert suggest("chitchat", {}) == []
        assert suggest("unknown", {}) == []

    def test_bounded(self):
        for intent in ("code_generation", "project_build", "design"):
            assert len(suggest(intent, {})) <= 3
