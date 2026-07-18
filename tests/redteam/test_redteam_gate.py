"""Adversarial red-team regression gate (roadmap Phase 1 #22 / gap-fill G1).

A data-driven battery of adversarial inputs run against the REAL, deterministic
security functions — prompt-injection detection, egress redaction, and the
misuse floor. No keys, no models, no network: safe for the offline CI gate.

Why this exists alongside the unit tests (test_safety.py / test_egress_redact.py):
those assert individual behaviors; this is a *growable corpus* treated as a gate,
and it uniquely locks the **misuse-floor-vs-over-refusal contract** — the safety
boundary the Uncensored Coding Mode feature depends on (block genuinely malicious
creation, but never over-refuse legitimate/authorized/defensive work).

Grow `corpus.json` over time; each entry is a locked contract.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.agent.safety import classify_misuse, has_injection
from app.security.egress_redact import redact_text

_CORPUS = json.loads(
    (pathlib.Path(__file__).resolve().parent / "corpus.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("text", _CORPUS["injection_expected_caught"])
def test_injection_is_detected(text):
    assert has_injection(text), (
        f"Prompt-injection NOT detected (regression): {text!r}. "
        f"scan_injection must flag this."
    )


@pytest.mark.parametrize("text", _CORPUS["benign_expected_clean"])
def test_benign_is_not_flagged(text):
    assert not has_injection(text), (
        f"False positive — benign text flagged as injection: {text!r}. "
        f"Injection patterns must not catch ordinary requests (or names like 'Dan')."
    )


@pytest.mark.parametrize("case", _CORPUS["secrets_expected_redacted"])
def test_secrets_are_redacted(case):
    _, n = redact_text(case["text"], mode=case["mode"])
    assert n > 0, (
        f"Secret NOT redacted (regression) in mode={case['mode']!r}: {case['text']!r}."
    )


@pytest.mark.parametrize("case", _CORPUS["pii_strict_expected_redacted"])
def test_strict_pii_is_redacted(case):
    _, n = redact_text(case["text"], mode=case["mode"])
    assert n > 0, f"Strict-mode PII NOT redacted (regression): {case['text']!r}."


@pytest.mark.parametrize("text", _CORPUS["misuse_expected_blocked"])
def test_malicious_creation_is_blocked(text):
    v = classify_misuse(text)
    assert v.blocked, (
        f"Misuse floor BREACHED (regression): {text!r} should be blocked. "
        f"This is the hard floor that survives even Uncensored Coding Mode."
    )


@pytest.mark.parametrize("text", _CORPUS["misuse_expected_allowed"])
def test_legitimate_security_work_is_allowed(text):
    v = classify_misuse(text)
    assert not v.blocked, (
        f"OVER-REFUSAL (regression): {text!r} is legitimate/authorized/defensive "
        f"and must NOT be blocked. This is the over-refusal-avoidance contract the "
        f"Uncensored Coding Mode depends on."
    )


@pytest.mark.parametrize("case", _CORPUS["known_limitations"])
@pytest.mark.xfail(reason="documented known limitation — see corpus note", strict=True)
def test_known_limitations_still_hold(case):
    # These document CURRENT gaps honestly. `strict=True` means if one starts
    # passing (i.e. the gap got fixed), THIS test fails — a signal to promote the
    # case into the real expectations above. So the corpus can never silently
    # drift out of date.
    if case["vector"] == "egress_redact":
        _, n = redact_text(case["text"], mode="secrets")
        assert n > 0
    else:  # pragma: no cover - future vectors
        assert has_injection(case["text"])


def test_corpus_is_substantial():
    # Guard against the corpus being gutted; a red-team gate with 2 cases is
    # theater. Keep it a real battery.
    total = sum(
        len(_CORPUS[k])
        for k in (
            "injection_expected_caught",
            "benign_expected_clean",
            "secrets_expected_redacted",
            "pii_strict_expected_redacted",
            "misuse_expected_blocked",
            "misuse_expected_allowed",
        )
    )
    assert total >= 40, f"Red-team corpus shrank to {total} cases — keep it broad."
