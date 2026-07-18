"""Tests for the clarification preference store (app/clarify/preferences).

Pure dict-based logic: in-conversation memory, durable promotion after repeats,
known-choice merging, analytics counters, and clear() (privacy).
"""
from __future__ import annotations

import pytest

_mod = pytest.importorskip("app.clarify.preferences")
Store = _mod.ClarificationPreferenceStore


def test_records_session_answer_and_known_choices():
    prefs: dict = {}
    s = Store(prefs, conversation_id="c1")
    s.record_answer("Language", "Python")
    assert s.session_prefs() == {"Language": "Python"}
    assert s.known_choices() == {"Language": "Python"}
    # persisted into the shared blob
    assert prefs["clarify"]["sessions"]["c1"]["Language"] == "Python"


def test_promotes_to_durable_after_three_repeats():
    prefs: dict = {}
    # Same value chosen across three different conversations.
    for cid in ("c1", "c2", "c3"):
        s = Store(prefs, conversation_id=cid)
        s.record_answer("Depth", "detailed")
    s = Store(prefs, conversation_id="c4")
    assert s.durable_prefs() == {"Depth": "detailed"}
    # durable applies even in a brand-new conversation
    assert s.known_choices() == {"Depth": "detailed"}


def test_does_not_promote_before_threshold():
    prefs: dict = {}
    for cid in ("c1", "c2"):
        Store(prefs, conversation_id=cid).record_answer("DB", "Postgres")
    assert Store(prefs).durable_prefs() == {}


def test_session_overrides_durable_in_known_choices():
    prefs: dict = {"clarify": {"durable": {"Language": "Python"}}}
    s = Store(prefs, conversation_id="c1")
    s.record_answer("Language", "Go")
    assert s.known_choices()["Language"] == "Go"


def test_record_answers_bulk_and_analytics():
    prefs: dict = {}
    s = Store(prefs, conversation_id="c1")
    s.record_answers({"Language": "Rust", "Platform": "CLI"})
    assert s.known_choices() == {"Language": "Rust", "Platform": "CLI"}
    assert s.analytics()["answered"] == 2


def test_analytics_record_increments_known_keys_only():
    prefs: dict = {}
    s = Store(prefs)
    s.analytics_record("asked")
    s.analytics_record("skipped", 2)
    s.analytics_record("bogus")  # ignored
    a = s.analytics()
    assert a["asked"] == 1 and a["skipped"] == 2
    assert "bogus" not in a


def test_mode_and_contract_roundtrip():
    prefs: dict = {}
    s = Store(prefs)
    s.set_mode("builder")
    s.set_contract({"autonomy": "recommend"})
    s.set_contract({"challenge": "sometimes"})
    assert s.mode() == "builder"
    assert s.contract() == {"autonomy": "recommend", "challenge": "sometimes"}


def test_clear_forgets_everything():
    prefs: dict = {}
    s = Store(prefs, conversation_id="c1")
    s.record_answer("Language", "Python")
    s.set_mode("expert")
    s.clear()
    assert s.durable_prefs() == {}
    assert s.session_prefs() == {}
    assert s.mode() is None
    assert s.analytics()["answered"] == 0


def test_ignores_blank_choice_or_value():
    prefs: dict = {}
    s = Store(prefs, conversation_id="c1")
    s.record_answer("", "x")
    s.record_answer("Y", "")
    assert s.known_choices() == {}


def test_backfills_missing_subkeys_on_old_blob():
    prefs: dict = {"clarify": {"durable": {"Language": "Java"}}}
    s = Store(prefs, conversation_id="c1")
    # missing sessions/counts/analytics are backfilled, durable preserved
    assert s.durable_prefs() == {"Language": "Java"}
    assert s.analytics()["answered"] == 0
    s.record_answer("Platform", "web")
    assert s.session_prefs() == {"Platform": "web"}


# ---- answer-line parser --------------------------------------------------

def test_parse_clarify_answer_lines():
    parse = _mod.parse_answer_lines
    text = "Language: Python\nFeatures: Auth, Chat\nPriority by priority: 1. Speed, 2. Cost"
    out = parse(text)
    assert out["Language"] == "Python"
    assert out["Features"] == "Auth, Chat"
    assert out["Priority"] == "1. Speed, 2. Cost"


def test_parse_ignores_prose_and_code_lines():
    parse = _mod.parse_answer_lines
    text = ("Here is a long sentence: that should be ignored because the head "
            "is prose\nprint('hello: world')")
    out = parse(text)
    assert out == {}
