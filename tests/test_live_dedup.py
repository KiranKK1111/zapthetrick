"""Near-duplicate question guard (live-latency/duplication report 2026-07-08).
Deterministic — stdlib similarity, injected clock."""
from __future__ import annotations

from app.live.dedup import QuestionDeduper, normalize


class TestNormalize:
    def test_strips_punct_case_filler(self):
        assert normalize("So, um... How would you SCALE Kafka?") == \
            normalize("how would you scale kafka")

    def test_empty_safe(self):
        assert normalize("") == ""


class TestDeduper:
    def _d(self):
        return QuestionDeduper(window_s=20.0, similarity=0.87)

    def test_exact_repeat_within_window_is_duplicate(self):
        d = self._d()
        d.note_answered("how would you scale kafka", now=0.0)
        assert d.is_duplicate("How would you scale Kafka?", now=5.0)

    def test_near_identical_retranscription_is_duplicate(self):
        d = self._d()
        d.note_answered("how would you scale kafka consumers", now=0.0)
        assert d.is_duplicate("um how would you scale the kafka consumers",
                              now=3.0)

    def test_repeat_after_window_answers_again(self):
        d = self._d()
        d.note_answered("how would you scale kafka", now=0.0)
        assert not d.is_duplicate("how would you scale kafka", now=25.0)

    def test_superset_continuation_passes(self):
        # Endpoint split: fragment answered, then the FULL question arrives —
        # the merge path must be allowed to answer the longer question.
        d = self._d()
        d.note_answered("so tell me", now=0.0)
        assert not d.is_duplicate(
            "so tell me how would you scale kafka", now=2.0)

    def test_different_questions_pass(self):
        d = self._d()
        d.note_answered("how would you scale kafka", now=0.0)
        assert not d.is_duplicate("what is a hash map", now=1.0)
        assert not d.is_duplicate(
            "how do you handle conflict in a team", now=1.0)

    def test_short_fragments_never_suppressed(self):
        d = self._d()
        d.note_answered("and why", now=0.0)
        assert not d.is_duplicate("and why", now=1.0)   # < 8 chars normalized

    def test_window_pruning(self):
        d = QuestionDeduper(window_s=5.0)
        d.note_answered("question one about databases", now=0.0)
        d.note_answered("question two about caching", now=4.0)
        assert d.is_duplicate("question two about caching", now=8.0)
        assert not d.is_duplicate("question one about databases", now=8.0)

    def test_fail_open_on_garbage(self):
        d = self._d()
        assert not d.is_duplicate(None)      # type: ignore[arg-type]
        d.note_answered(None)                # type: ignore[arg-type]


class TestSessionCounts:
    """Per-session ledger counters feed the health frame (enhancement #3)."""

    def setup_method(self):
        from app.live import ledger
        ledger.reset_for_tests()

    def teardown_method(self):
        from app.live import ledger
        ledger.reset_for_tests()

    def test_counts_per_session_with_reason(self):
        from app.live import ledger
        ledger.record("s1", "q1", "how would you scale kafka",
                      ledger.SKIPPED, reason="duplicate_question")
        ledger.record("s1", "q2", "what is a hash map", ledger.ANSWERED)
        ledger.record("s2", "q3", "tell me about yourself", ledger.ANSWERED)
        c1 = ledger.session_counts("s1")
        assert c1.get("skipped:duplicate_question") == 1
        assert c1.get("answered") == 1
        assert ledger.session_counts("s2").get("answered") == 1
        assert "skipped:duplicate_question" not in ledger.session_counts("s2")

    def test_unknown_session_empty(self):
        from app.live import ledger
        assert ledger.session_counts("nope") == {}

    def test_reset_clears_session_counts(self):
        from app.live import ledger
        ledger.record("s1", "q1", "x y z question", ledger.ANSWERED)
        ledger.reset_for_tests()
        assert ledger.session_counts("s1") == {}

    def test_fifo_bound(self):
        from app.live import ledger
        for i in range(70):
            ledger.record(f"sid{i}", None, "some question here",
                          ledger.ANSWERED)
        # Oldest sessions pruned; the newest is retained.
        assert ledger.session_counts("sid0") == {}
        assert ledger.session_counts("sid69").get("answered") == 1


class TestLiveEnhancementConfig:
    """The 2026-07-08 enhancement flags must exist on LiveSection with the
    right defaults — a missing field means yaml values get silently dropped
    (the candidate_echo_skip bug)."""

    def test_defaults(self):
        from app.core.config_loader import LiveSection
        s = LiveSection()
        assert s.question_dedup is True
        assert s.question_dedup_window_s == 20.0
        assert s.question_dedup_similarity == 0.87
        assert s.candidate_echo_skip is True
        assert s.candidate_echo_threshold == 0.72
        assert s.combine_multi_questions is True
        assert s.factual_fast_path is True
