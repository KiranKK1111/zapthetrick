"""Tests for the Response Fingerprint (roadmap Phase 6 #22), including that it's
wired into the response.v1 envelope's meta.
"""
from __future__ import annotations

from app.response_arch.envelope import build_envelope
from app.response_arch.fingerprint import content_hash, response_fingerprint


def test_content_hash_is_whitespace_stable():
    assert content_hash("hello   world") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")


def test_fingerprint_is_deterministic():
    a = response_fingerprint(content="answer", model="m1", app_version="0.2.0")
    b = response_fingerprint(content="answer", model="m1", app_version="0.2.0")
    assert a == b and a["hash"]


def test_fingerprint_changes_with_provenance():
    base = response_fingerprint(content="answer", model="m1", app_version="0.2.0")
    diff_model = response_fingerprint(content="answer", model="m2", app_version="0.2.0")
    diff_content = response_fingerprint(content="other", model="m1", app_version="0.2.0")
    # Same text, different model -> different fingerprint (why did it change?).
    assert diff_model["hash"] != base["hash"]
    assert diff_content["hash"] != base["hash"]
    # ...but content_hash is stable across the model change.
    assert diff_model["content_hash"] == base["content_hash"]


def test_sources_order_independent():
    a = response_fingerprint(content="x", knowledge_sources=["resume", "web"])
    b = response_fingerprint(content="x", knowledge_sources=["web", "resume"])
    assert a["hash"] == b["hash"]


def test_fingerprint_fail_open():
    assert response_fingerprint(content=None) is not None  # type: ignore[arg-type]


def test_wired_into_envelope_meta():
    env = build_envelope(model="test-model", resolved_prompt="explain kafka",
                         content="Kafka is a distributed log.")
    fp = env["meta"]["fingerprint"]
    assert fp["hash"] and fp["model"] == "test-model"
    assert "app_version" in fp  # provenance from Phase 1 APP_VERSION


def test_envelope_still_builds_without_content():
    # Backwards compatible: existing callers that don't pass content still work.
    env = build_envelope(model="m", resolved_prompt="hi")
    assert env["meta"]["fingerprint"]["hash"]
