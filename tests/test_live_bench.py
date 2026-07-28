"""Gap 1 — live question-detection harness runs on the annotated corpus and
meets the report's accuracy/false-answer targets. Doubles as a regression gate:
a detection change that drops accuracy or spikes false-answers fails here."""
from app.eval import live_bench


def test_harness_runs_and_reports_all_metrics():
    r = live_bench.run_corpus()
    for k in ("accuracy", "precision", "recall", "f1", "false_answer_rate",
              "multi_question_recall", "fast_path_coverage", "counts"):
        assert k in r, k
    assert r["total"] >= 20


def test_detection_accuracy_targets_deterministic_path(monkeypatch):
    """The DETERMINISTIC path only (semantic gates forced off) — i.e. cold start,
    before the embedder finishes loading. Pinned explicitly so the result can't
    drift with test ORDER (whether some earlier test happened to warm the
    embedder), which is how a real false-answer bug once hid here.

    The bar is honestly LOWER than the production bar below: literal cue/prefix
    lists cannot separate "give me one moment, my screen froze" from "give me an
    example", so a couple of interviewer-floor utterances still promote until the
    semantic veto is available. Fail-open is deliberate — never worse than before
    the gates existed — and the pod loads the embedder at startup, so this window
    is brief."""
    from app.semantics import gates
    monkeypatch.setattr(gates, "_enabled", lambda: False)
    r = live_bench.run_corpus()
    assert r["accuracy"] >= 0.9, r["failures"]
    assert r["recall"] >= 0.9, r["failures"]
    assert r["precision"] >= 0.85, r["failures"]
    assert r["multi_question_recall"] >= 0.9
    # Guard the known ceiling so a NEW deterministic false-answer still fails.
    assert r["false_answer_rate"] <= 0.14, r["failures"]


def test_detection_targets_hold_with_the_semantic_embedder_live():
    """PRODUCTION-STATE check. Detection also runs a semantic implicit-question
    gate, which is INERT until the embedder is loaded — so the plain run above
    only exercises the deterministic path. On the pod the embedder IS live, and
    it once promoted the interviewer's own "Let me tell you a bit about the team"
    to a question (a false answer talking over their intro). Warm the embedder
    and assert the targets still hold, so that regression can't return silently.
    Skips when no embedder is available (slim/CI env)."""
    import pytest
    try:
        from app.rag import embedder
        embedder.embed(["warm up"])
        if not embedder.is_ready():
            pytest.skip("embedder unavailable")
    except Exception:  # noqa: BLE001
        pytest.skip("embedder unavailable")
    r = live_bench.run_corpus()
    assert r["false_answer_rate"] < 0.05, r["failures"]
    assert r["precision"] >= 0.9, r["failures"]
    assert r["recall"] >= 0.9, r["failures"]


def test_fast_path_coverage_is_measured():
    # Latency signal — informational, but must be a real ratio in [0,1]. (It is
    # LOW today: many correct answers still pay the slow LLM round-trip — the
    # documented tuning opportunity the harness exists to surface.)
    r = live_bench.run_corpus()
    assert 0.0 <= r["fast_path_coverage"] <= 1.0
