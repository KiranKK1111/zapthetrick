"""
Phase 16 tests — out-of-band extensions
(live-conversational-intelligence R60, R61, R62).

Property 60: STT device resolution selects GPU only when enabled AND CUDA is
present, else falls back cleanly to CPU; transcription semantics are unchanged.
Property 61: career intelligence is advisory, disabled by default, and returns
nothing meaningful on error / empty input.
Property 62: enterprise readiness isolates per-user data (no cross-user leakage),
aggregates team analytics PII-free, and is a no-op when disabled (single-user
unchanged).
"""
from __future__ import annotations

from app.live import career as _career
from app.live import tenancy as _ten
from app.live.org import build_org
from app.live.profile import build_profile


# ---- R60: GPU STT device selection -----------------------------------------

def test_stt_device_falls_back_to_cpu_without_cuda(monkeypatch):
    from app.stt import factory as _factory
    # Force "no CUDA" and flag on → must fall back to the configured CPU device.
    monkeypatch.setattr(_factory, "_cuda_available", lambda: False)
    monkeypatch.setattr(_factory.cfg.live, "gpu_stt", True, raising=False)
    device, ctype = _factory.resolve_device()
    assert device == "cpu"
    # int8 (a CPU compute type) preserved on fallback.
    assert ctype in ("int8", "int8_float16", _factory.cfg.stt.compute_type)


def test_stt_device_selects_gpu_when_available_and_enabled(monkeypatch):
    from app.stt import factory as _factory
    monkeypatch.setattr(_factory, "_cuda_available", lambda: True)
    monkeypatch.setattr(_factory.cfg.live, "gpu_stt", True, raising=False)
    device, ctype = _factory.resolve_device()
    assert device == "cuda"
    assert ctype == "float16"   # GPU compute type, not int8


def test_stt_device_cpu_when_flag_off(monkeypatch):
    from app.stt import factory as _factory
    monkeypatch.setattr(_factory, "_cuda_available", lambda: True)
    monkeypatch.setattr(_factory.cfg.live, "gpu_stt", False, raising=False)
    device, _ = _factory.resolve_device()
    assert device == "cpu"   # flag off → no GPU even when available


# ---- R61: advisory career intelligence -------------------------------------

def test_career_is_advisory_with_disclaimer():
    prof = build_profile({"skills": ["python", "kafka"],
                          "projects": [{"name": "Billing", "tech": ["redis"]}],
                          "achievements": ["cut latency 40%"]})
    org = build_org("Acme", "We need python and kubernetes.", "Backend")
    from app.live.org import fit_analysis
    fit = fit_analysis(prof, org)
    ci = _career.analyze(prof, org, fit=fit)
    assert ci.advisory is True
    assert "NOT professional" in ci.disclaimer
    assert "kubernetes" in ci.skill_gaps   # JD skill the candidate lacks
    assert ci.readiness in ("entry", "mid", "senior_ready")


def test_career_empty_on_no_profile():
    ci = _career.analyze(None)
    assert ci.coaching == []
    assert ci.readiness == "unknown"
    assert ci.advisory is True


# ---- R62: enterprise readiness ---------------------------------------------

def test_owns_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(_ten, "is_enabled", lambda: False)
    # Disabled → single-user → everything is "owned" (no scoping).
    assert _ten.owns("user-A", "user-B") is True


def test_owns_isolates_users_when_enabled(monkeypatch):
    monkeypatch.setattr(_ten, "is_enabled", lambda: True)
    assert _ten.owns("user-A", "user-A") is True
    assert _ten.owns("user-A", "user-B") is False   # no cross-user access
    assert _ten.owns(None, "user-A") is False


def test_scope_query_filters_when_enabled(monkeypatch):
    monkeypatch.setattr(_ten, "is_enabled", lambda: True)
    records = [{"owner": "u1", "x": 1}, {"owner": "u2", "x": 2}, {"owner": "u1", "x": 3}]
    out = _ten.scope_query(records, "u1")
    assert {r["x"] for r in out} == {1, 3}
    # Disabled → unchanged.
    monkeypatch.setattr(_ten, "is_enabled", lambda: False)
    assert len(_ten.scope_query(records, "u1")) == 3


def test_strip_pii_and_aggregation_is_pii_free():
    rec = {"owner": "u1", "email": "a@b.com", "name": "Alice", "answers": 5,
           "topics": ["kafka"]}
    clean = _ten.strip_pii(rec)
    assert "email" not in clean and "name" not in clean
    assert clean["answers"] == 5
    ta = _ten.aggregate_team([
        {"email": "a@b.com", "answers": 4, "topics": ["kafka", "redis"]},
        {"name": "Bob", "answers": 6, "topics": ["kafka"]},
    ])
    d = ta.to_dict()
    assert d["pii_free"] is True
    assert d["sessions"] == 2
    assert d["answers"] == 10
    assert "kafka" in d["top_topics"]
