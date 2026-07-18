"""Contract tests for the Clarifier's JSON parsing + sanitisation
(app/agents/clarifier._parse_questions).

The clarifier asks only when it should and shapes questions the UI relies on.
The model call isn't unit-testable, but the parse/decline/kind logic is pure and
is where a bad reply would otherwise surface a wrong or empty clarify panel.
"""
from __future__ import annotations

import json

import pytest

_mod = pytest.importorskip("app.agents.clarifier")
_parse_questions = _mod._parse_questions
language_choice_question = _mod.language_choice_question
document_choice_question = _mod.document_choice_question
archive_format_question = _mod.archive_format_question
_has_doc_signal = _mod._has_doc_signal
_has_code_request = _mod._has_code_request


def _dump(obj) -> str:
    return json.dumps(obj)


def test_declines_when_clarify_false():
    assert _parse_questions(_dump({"clarify": False, "questions": []})) == []


def test_language_choice_question_shape():
    qs = language_choice_question()
    assert len(qs) == 1
    q = qs[0]
    assert q["header"] == "Language" and q["kind"] == "single"
    labels = [o["label"] for o in q["options"]]
    assert "Python" in labels and any("decide" in l.lower() for l in labels)
    assert sum(o["recommended"] for o in q["options"]) == 1


def test_document_choice_question_shape():
    qs = document_choice_question()
    assert len(qs) == 1
    q = qs[0]
    assert q["header"] == "Format" and q["kind"] == "single"
    assert 2 <= len(q["options"]) <= 4
    assert sum(o["recommended"] for o in q["options"]) == 1
    # Offers the common export formats so the user picks one (not a silent PDF).
    ids = {o["id"] for o in q["options"]}
    assert {"pdf", "docx"} <= ids


class TestDocSignalGate:
    """The 'which document format?' clarification must fire ONLY when the
    message actually signals a file deliverable. Regression (2026-07-14): a
    plain follow-up — "can you give me some more details on it" — wrongly
    triggered the format card (the LLM assessment guessed a doc_format slot),
    and choosing Word then produced no document. The deterministic gate stops
    the false ask."""

    @pytest.mark.parametrize("text", [
        "can you give me some more details on it",
        "give me some more details on it",
        "what is this error in the screenshot",
        "explain it a bit more",
        "why does this happen",
        # Display / rendering-format asks are NOT document requests — they
        # describe how the inline answer should look (regression: "tabular
        # format" fired the "which document format?" card).
        "can you get me in a tabular format",
        "show it as a table",
        "give me this as bullet points",
    ])
    def test_no_signal_on_plain_followups(self, text):
        assert _has_doc_signal(text) is False

    @pytest.mark.parametrize("text", [
        "give me this as a word document",
        "can you make this downloadable",
        "put this in a pdf",
        "export this conversation",
        "i need a soft copy of this",
        "can you put this into a document",
    ])
    def test_signal_on_real_deliverables(self, text):
        assert _has_doc_signal(text) is True


class TestCodeRequestGate:
    """The 'which language?' clarification must fire ONLY on a genuine
    write-code request. Regression (2026-07-16): the intent classifier
    mislabelled a statement ("I don't want pin and section") and a rendering
    follow-up ("can you get me in a tabular format") as code_generation, and
    the language card fired with no evidence gate. Authority is the semantic
    `code_request` gate; these assertions hold on the deterministic fail-open
    fast-path too (the embedder is stubbed in CI)."""

    @pytest.mark.parametrize("text", [
        "I don't want pin and section",
        "can you get me in a tabular format",
        "show it as a table",
        "explain how kafka works",
        "summarize the above",
        "give me more details on it",
        "what is the difference between monolith and microservices",
    ])
    def test_no_language_ask_without_code_request(self, text):
        assert _has_code_request(text) is False

    # These are recognised by the deterministic fail-open fast-path (verb + code
    # noun), so the assertion holds in CI where the embedder is stubbed. The
    # semantic gate additionally catches paraphrases like "implement quicksort".
    @pytest.mark.parametrize("text", [
        "write a program to reverse a string",
        "create a python script to read a csv",
        "build a rest api for orders",
        "implement a binary search function",
    ])
    def test_language_ask_on_real_code_request(self, text):
        assert _has_code_request(text) is True


def test_archive_format_question_shape():
    qs = archive_format_question()
    assert len(qs) == 1
    q = qs[0]
    assert q["header"] == "Format" and q["kind"] == "single"
    labels = [o["label"].lower() for o in q["options"]]
    assert any("zip" in l for l in labels) and any("7z" in l for l in labels)
    # RAR is intentionally NOT offered (can't be created with open tooling).
    assert not any("rar" in l for l in labels)
    assert sum(o["recommended"] for o in q["options"]) == 1


def test_declines_on_garbage():
    assert _parse_questions("") == []
    assert _parse_questions("not json") == []
    assert _parse_questions(_dump({"questions": "nope"})) == []


def test_parses_a_valid_question():
    raw = _dump({"clarify": True, "questions": [{
        "question": "Which language?", "header": "Language", "kind": "multi",
        "options": [{"label": "Python", "description": "concise"},
                    {"label": "Go", "description": "fast"}],
    }]})
    out = _parse_questions(raw)
    assert len(out) == 1
    q = out[0]
    assert q["question"] == "Which language?"
    assert q["kind"] == "multi" and q["multiSelect"] is True
    assert [o["label"] for o in q["options"]] == ["Python", "Go"]


def test_model_kind_is_trusted_not_overridden():
    # A "language" question explicitly marked single stays single — no keyword
    # promotion second-guesses the model anymore.
    raw = _dump({"clarify": True, "questions": [{
        "question": "Which language for the runtime?", "header": "Lang",
        "kind": "single",
        "options": [{"label": "Python", "description": ""},
                    {"label": "Node", "description": ""}],
    }]})
    out = _parse_questions(raw)
    assert out[0]["kind"] == "single" and out[0]["multiSelect"] is False


def test_missing_kind_falls_back_to_multiselect_bool():
    raw = _dump({"clarify": True, "questions": [{
        "question": "Pick features", "header": "Features", "multiSelect": True,
        "options": [{"label": "A"}, {"label": "B"}],
    }]})
    assert _parse_questions(raw)[0]["kind"] == "multi"


def test_question_with_under_two_options_is_dropped():
    raw = _dump({"clarify": True, "questions": [{
        "question": "Only one?", "options": [{"label": "X"}],
    }]})
    assert _parse_questions(raw) == []


def test_caps_questions_and_options_and_header():
    raw = _dump({"clarify": True, "questions": [
        {"question": f"Q{i}", "header": "x" * 40,
         "options": [{"label": f"o{j}"} for j in range(8)]}
        for i in range(6)
    ]})
    out = _parse_questions(raw)
    assert len(out) <= 3                      # _MAX_QUESTIONS
    assert all(len(q["options"]) <= 4 for q in out)   # _MAX_OPTIONS
    assert all(len(q["header"]) <= 16 for q in out)


# ---------------------------------------------------------------------------
# Phase 1 — payload parsing (confidence / blocking / reason / mode / ids)
# ---------------------------------------------------------------------------

_parse_payload = _mod._parse_payload
_apply_band = _mod._apply_band
is_sample_popup_request = _mod.is_sample_popup_request


def test_payload_defaults_on_decline():
    qs, meta = _parse_payload(_dump({"clarify": False}))
    assert qs == []
    assert meta["confidence"] == 1.0
    assert meta["blocking"] is False
    assert meta["reason"] == ""
    assert meta["estimated_questions_saved"] == 0
    assert meta["mode"] == "ask"
    assert meta["assumptions"] == []


def test_payload_extracts_meta_fields():
    raw = _dump({
        "clarify": True, "confidence": 0.45, "blocking": True,
        "reason": "It changes the design.", "estimated_questions_saved": 4,
        "mode": "ask",
        "questions": [{
            "question": "Which language?", "header": "Lang", "kind": "single",
            "reason": "drives the codebase",
            "options": [{"label": "Python", "description": "x"},
                        {"label": "Go", "description": "y"}],
        }],
    })
    qs, meta = _parse_payload(raw)
    assert meta["confidence"] == 0.45 and meta["blocking"] is True
    assert meta["reason"] == "It changes the design."
    assert meta["estimated_questions_saved"] == 4
    assert qs[0]["id"] == "q1" and qs[0]["reason"] == "drives the codebase"
    assert qs[0]["options"][0]["id"] == "o1"


def test_confidence_out_of_range_defaults():
    # declining + bad confidence → 1.0
    _, meta = _parse_payload(_dump({"clarify": False, "confidence": 9}))
    assert meta["confidence"] == 1.0
    # asking + missing confidence → mid (keeps the ask)
    _, meta2 = _parse_payload(_dump({
        "clarify": True,
        "questions": [{"question": "Q", "options": [{"label": "A"}, {"label": "B"}]}],
    }))
    assert meta2["confidence"] == 0.5


def test_estimated_saved_clamped():
    _, meta = _parse_payload(_dump({"clarify": False,
                                    "estimated_questions_saved": 999}))
    assert meta["estimated_questions_saved"] == 99


def test_assume_mode_parses_assumptions():
    raw = _dump({
        "clarify": True, "confidence": 0.8, "mode": "assume",
        "assumptions": [{"label": "Frontend", "value": "Flutter"},
                        {"label": "Backend", "value": "Python"}],
        "questions": [],
    })
    _, meta = _parse_payload(raw)
    assert meta["mode"] == "assume"
    assert [a["value"] for a in meta["assumptions"]] == ["Flutter", "Python"]
    assert meta["assumptions"][0]["id"] == "a1"


def test_apply_band_drops_on_high_confidence():
    qs = [{"id": "q1"}, {"id": "q2"}]
    assert _apply_band(qs, {"confidence": 0.95, "mode": "ask"}) == []


def test_apply_band_caps_targeted_to_two():
    qs = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
    assert len(_apply_band(qs, {"confidence": 0.5, "mode": "ask"})) == 2


def test_apply_band_assumption_band_caps_to_one():
    qs = [{"id": "q1"}, {"id": "q2"}]
    assert len(_apply_band(qs, {"confidence": 0.8, "mode": "ask"})) == 1


def test_apply_band_guided_keeps_up_to_three():
    qs = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
    assert len(_apply_band(qs, {"confidence": 0.2, "mode": "ask"})) == 3


def test_sample_popup_detection():
    assert is_sample_popup_request("can you provide a sample user question popup")
    assert is_sample_popup_request("show me an example clarification popup")
    assert not is_sample_popup_request("build me a todo app")


def test_default_build_questions_have_ids():
    qs = _mod.default_build_questions()
    assert all(q.get("id") for q in qs)
    assert all(all(o.get("id") for o in q["options"]) for q in qs)


def test_at_most_one_recommended_option_preserved():
    raw = _dump({"clarify": True, "confidence": 0.5, "questions": [{
        "question": "Which db?", "header": "DB", "kind": "single",
        "options": [
            {"label": "Postgres", "recommended": True},
            {"label": "MySQL", "recommended": True},
            {"label": "SQLite", "recommended": True},
        ],
    }]})
    out = _parse_questions(raw)
    recs = [o for o in out[0]["options"] if o["recommended"]]
    assert len(recs) == 1 and recs[0]["label"] == "Postgres"


def test_per_question_reason_is_captured_and_capped():
    long_reason = "x" * 400
    raw = _dump({"clarify": True, "confidence": 0.5, "questions": [{
        "question": "Q", "header": "H", "kind": "single", "reason": long_reason,
        "options": [{"label": "A"}, {"label": "B"}],
    }]})
    out = _parse_questions(raw)
    assert len(out[0]["reason"]) == 200


def test_preview_flag_parsed_and_defaults_false():
    _, meta = _parse_payload(_dump({"clarify": False}))
    assert meta["preview"] is False
    _, meta2 = _parse_payload(_dump({
        "clarify": True, "confidence": 0.4, "preview": True,
        "questions": [{"question": "Q", "options": [{"label": "A"}, {"label": "B"}]}],
    }))
    assert meta2["preview"] is True


# ---------------------------------------------------------------------------
# Phase 5 — adaptive mode clamping (_apply_mode)
# ---------------------------------------------------------------------------

_apply_mode = _mod._apply_mode


def test_mode_autopilot_drops_unless_low_confidence():
    qs = [{"id": "q1"}, {"id": "q2"}]
    assert _apply_mode(qs, {"confidence": 0.5}, "autopilot") == []
    assert len(_apply_mode(qs, {"confidence": 0.2}, "autopilot")) == 2


def test_mode_builder_caps_to_one():
    qs = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
    assert len(_apply_mode(qs, {"confidence": 0.3}, "builder")) == 1


def test_mode_explorer_keeps_band_count():
    qs = [{"id": "q1"}, {"id": "q2"}]
    assert len(_apply_mode(qs, {"confidence": 0.3}, "explorer")) == 2


def test_mode_none_or_empty_is_noop():
    qs = [{"id": "q1"}, {"id": "q2"}]
    assert _apply_mode(qs, {"confidence": 0.3}, "") == qs
    assert _apply_mode([], {"confidence": 0.3}, "builder") == []
