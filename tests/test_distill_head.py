"""Tests for distilled decision heads (vNext §9.7, Stage 11 Component B)."""
from __future__ import annotations

import app.eval.flywheel as F
import app.llm.distill_head as D


def _on(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: True)


def _head(target=D.SKIP_ANSWER, version="v1"):
    return D.DistillHead(target=target, version=version, labels=("skip", "answer"))


# ---- DistillHead.predict --------------------------------------------------
def test_predict_top_label():
    p = _head().predict([0.1], model_fn=lambda e: {"answer": 0.8, "skip": 0.2})
    assert p.label == "answer" and p.version == "v1"
    assert 0 < p.confidence <= 1


def test_predict_filters_unknown_labels():
    p = _head().predict([0.1], model_fn=lambda e: {"bogus": 0.9, "skip": 0.1})
    assert p.label == "skip"                      # bogus dropped


def test_predict_no_model_is_none():
    assert _head().predict([0.1], model_fn=None) is None


def test_predict_never_raises():
    assert _head().predict([0.1], model_fn=lambda e: 1 / 0) is None


# ---- promotion gate (never worse) -----------------------------------------
def test_promote_only_on_promote_verdict():
    reg = D.HeadRegistry()
    assert reg.consider(_head(), verdict=F.HOLD) is False
    assert reg.consider(_head(), verdict=F.ROLLBACK) is False
    assert reg.consider(_head(version="v2"), verdict=F.PROMOTE) is True
    assert reg.active_version(D.SKIP_ANSWER) == "v2"


def test_promote_rejects_unknown_target():
    reg = D.HeadRegistry()
    assert reg.consider(D.DistillHead("bogus_target", "v1"), verdict=F.PROMOTE) is False


def test_rollback_removes_active_head(monkeypatch):
    _on(monkeypatch)
    reg = D.HeadRegistry()
    reg.consider(_head(), verdict=F.PROMOTE)
    reg.rollback(D.SKIP_ANSWER)
    d = reg.decide(D.SKIP_ANSWER, [0.1],
                   model_fn=lambda e: {"answer": 0.99, "skip": 0.01},
                   fallback_fn=lambda e: "OLD")
    assert d.source == D.FALLBACK and d.label == "OLD"


# ---- front-of-path decide -------------------------------------------------
def test_confident_head_decides(monkeypatch):
    _on(monkeypatch)
    reg = D.HeadRegistry()
    reg.consider(_head(version="v5"), verdict=F.PROMOTE)
    d = reg.decide(D.SKIP_ANSWER, [0.1],
                   model_fn=lambda e: {"answer": 0.9, "skip": 0.1},
                   fallback_fn=lambda e: "OLD")
    assert d.source == D.DISTILLED and d.label == "answer" and d.version == "v5"


def test_unsure_head_falls_back(monkeypatch):
    _on(monkeypatch)
    reg = D.HeadRegistry()
    reg.consider(_head(), verdict=F.PROMOTE)
    d = reg.decide(D.SKIP_ANSWER, [0.1],
                   model_fn=lambda e: {"answer": 0.55, "skip": 0.45},
                   fallback_fn=lambda e: "OLD", min_confidence=0.7)
    assert d.source == D.FALLBACK and d.label == "OLD"


def test_no_promoted_head_falls_back(monkeypatch):
    _on(monkeypatch)
    d = D.HeadRegistry().decide(D.INTENT, [0.1], fallback_fn=lambda e: "OLD")
    assert d.source == D.FALLBACK and "no promoted head" in d.reason


def test_disabled_always_falls_back(monkeypatch):
    monkeypatch.setattr(D, "enabled", lambda: False)
    reg = D.HeadRegistry()
    reg.consider(_head(), verdict=F.PROMOTE)
    d = reg.decide(D.SKIP_ANSWER, [0.1],
                   model_fn=lambda e: {"answer": 0.99, "skip": 0.01},
                   fallback_fn=lambda e: "OLD")
    assert d.source == D.FALLBACK


def test_decide_logs_version(monkeypatch):
    _on(monkeypatch)
    reg = D.HeadRegistry()
    reg.consider(_head(version="v9"), verdict=F.PROMOTE)
    reg.decide(D.SKIP_ANSWER, [0.1],
               model_fn=lambda e: {"answer": 0.95, "skip": 0.05})
    assert (D.SKIP_ANSWER, "v9") in reg.log


def test_decide_fallback_error_is_safe(monkeypatch):
    _on(monkeypatch)
    reg = D.HeadRegistry()
    reg.consider(_head(), verdict=F.PROMOTE)
    # A model that returns nothing → fallback; the fallback also errors → empty.
    d = reg.decide(D.SKIP_ANSWER, [0.1], model_fn=lambda e: {},
                   fallback_fn=lambda e: 1 / 0)
    assert d.source == D.FALLBACK and d.label == ""


def test_target_order_is_promotion_order():
    assert D.TARGET_ORDER[0] == D.SKIP_ANSWER      # highest-value target first
    assert D.TARGET_ORDER == (D.SKIP_ANSWER, D.INTENT, D.DIFFICULTY,
                              D.FRESHNESS, D.GATES)


# ---- training-set builder -------------------------------------------------
def test_build_training_set_from_golden(monkeypatch):
    monkeypatch.setattr(F, "flywheel_enabled", lambda: True)
    gs = F.GoldenSet()
    gs.add(F.capture_case(F.THUMBS_DOWN, prompt="what is kafka"))
    gs.add(F.capture_case(F.VERIFY_FAIL, prompt="fix this bug"))
    ts = D.build_training_set(gs, D.INTENT,
                              label_fn=lambda c: "question" if "what" in c.prompt else "task")
    labels = sorted(x["label"] for x in ts)
    assert labels == ["question", "task"]


def test_build_training_set_skips_unlabeled(monkeypatch):
    monkeypatch.setattr(F, "flywheel_enabled", lambda: True)
    gs = F.GoldenSet()
    gs.add(F.capture_case(F.THUMBS_DOWN, prompt="a"))
    gs.add(F.capture_case(F.VERIFY_FAIL, prompt="b"))
    ts = D.build_training_set(gs, D.INTENT,
                              label_fn=lambda c: "x" if c.prompt == "a" else None)
    assert len(ts) == 1 and ts[0]["prompt"] == "a"


def test_build_training_set_never_raises():
    assert D.build_training_set(None, D.INTENT, label_fn=lambda c: "x") == []
