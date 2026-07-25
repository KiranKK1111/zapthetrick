"""Idempotency registry for composed sends (vNext §3.4)."""
from __future__ import annotations

from app.api import idempotency as I


def setup_function():
    I.reset_for_tests()


def test_first_claim_is_new():
    assert I.claim("k1") == (True, None)


def test_duplicate_while_in_flight_returns_false_none():
    I.claim("k1")
    is_new, prior = I.claim("k1")     # double-tap mid-stream
    assert is_new is False
    assert prior is None


def test_duplicate_after_complete_returns_the_result():
    I.claim("k1")
    I.complete("k1", {"message_id": "m-42", "user_message_id": "u-1"})
    is_new, prior = I.claim("k1")     # retry after it finished
    assert is_new is False
    assert prior == {"message_id": "m-42", "user_message_id": "u-1"}


def test_release_lets_a_genuine_retry_proceed():
    I.claim("k1")
    I.release("k1")                   # first turn errored before completing
    assert I.claim("k1") == (True, None)


def test_release_does_not_wipe_a_completed_key():
    I.claim("k1")
    I.complete("k1", {"message_id": "m-1"})
    I.release("k1")                   # no-op on a completed key
    is_new, prior = I.claim("k1")
    assert is_new is False
    assert prior == {"message_id": "m-1"}


def test_empty_key_is_always_new_and_never_stored():
    assert I.claim("") == (True, None)
    assert I.claim(None) == (True, None)
    I.complete("", {"x": 1})          # no-op
    assert I.claim("") == (True, None)


def test_distinct_keys_are_independent():
    assert I.claim("a")[0] is True
    assert I.claim("b")[0] is True
    assert I.claim("a")[0] is False
