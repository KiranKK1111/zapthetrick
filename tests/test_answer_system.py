"""Stage-7 §4.13 — technical answer system: role lens, depth ladder, envelope."""
from __future__ import annotations

import pytest

from app.live import answer_system as A


@pytest.fixture(autouse=True)
def _fresh():
    A.reset_for_tests()
    yield
    A.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.live, "answer_system", True, raising=False)


class TestRoleLens:
    def test_backend_detected(self):
        assert A.detect_role("Senior Backend Engineer",
                             "REST API, Kafka, Postgres", ["gRPC"]) == "backend"

    def test_frontend_detected(self):
        assert A.detect_role("Frontend Dev", "React, CSS, accessibility") \
            == "frontend"

    def test_sre_from_jd_only(self):
        assert A.detect_role("", "kubernetes on-call incident monitoring") == "sre"

    def test_ml_and_security(self):
        assert A.detect_role("", "pytorch model training inference") == "ml"
        assert A.detect_role("", "pentest vulnerability oauth threat") == "security"

    def test_generalist_when_nothing_scores(self):
        assert A.detect_role("Engineer", "work hard on cool things") == "generalist"
        assert A.detect_role("") == "generalist"

    def test_role_directive_carries_angle(self):
        assert "data modeling" in A.role_directive("backend")
        assert "generalist" in A.role_directive("").lower()


class TestDepthLadder:
    def test_starts_at_l1(self):
        assert A.depth("s1", "kafka") == A.L1

    def test_advances_and_never_restarts(self):
        assert A.advance("s1", "kafka") == A.L2
        assert A.advance("s1", "kafka") == A.L3
        assert A.depth("s1", "kafka") == A.L3        # persists (no restart)

    def test_caps_at_l4(self):
        for _ in range(10):
            A.advance("s1", "kafka")
        assert A.depth("s1", "kafka") == A.L4

    def test_per_topic_and_session(self):
        A.advance("s1", "kafka")
        assert A.depth("s1", "redis") == A.L1        # different topic
        assert A.depth("s2", "kafka") == A.L1        # different session

    def test_depth_directive_says_go_deeper(self):
        d = A.depth_directive(A.L3)
        assert "L3" in d and "deeper" in d.lower() and "restart" in d.lower()

    def test_set_depth_clamps(self):
        A.set_depth("s1", "x", 9)
        assert A.depth("s1", "x") == A.L4
        A.set_depth("s1", "x", -1)
        assert A.depth("s1", "x") == A.L1

    def test_forget_session(self):
        A.advance("s1", "kafka")
        A.forget_session("s1")
        assert A.depth("s1", "kafka") == A.L1


class TestEnvelope:
    def test_in_envelope_claim(self):
        assert A.in_envelope("I built a Kafka pipeline",
                             ["kafka pipeline", "python"]) is True

    def test_out_of_envelope_claim(self):
        assert A.in_envelope("I trained a transformer from scratch",
                             ["kafka pipeline", "python"]) is False

    def test_empty_envelope_is_out(self):
        # No resume facts → nothing is claimable → honest-frame everything.
        assert A.in_envelope("I did anything", []) is False

    def test_honest_frame_directive(self):
        d = A.honest_frame_directive()
        assert "HONESTLY" in d and "never claim direct experience" in d

    def test_unknown_directive(self):
        d = A.unknown_directive()
        assert "FIRST PRINCIPLES" in d and "assumptions" in d.lower()


class TestFlag:
    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "answer_system", False, raising=False)
        assert A.enabled() is False

    def test_enabled_reads_flag(self, _on):
        assert A.enabled() is True
