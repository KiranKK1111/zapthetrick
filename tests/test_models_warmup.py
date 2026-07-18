"""Startup model warm-up status registry (drives the download modal)."""
from __future__ import annotations

from app import models_warmup as mw


def _reset():
    mw.reset_for_tests()


def test_empty_is_ready():
    # Nothing registered → all_ready True so the UI never gets stuck.
    _reset()
    s = mw.snapshot()
    assert s["all_ready"] is True
    assert s["total"] == 0
    assert s["percent"] == 100.0


def test_lifecycle_pending_loading_ready():
    _reset()
    mw.register("embedder", "bge-m3")
    mw.register("stt:parakeet", "Parakeet")
    s = mw.snapshot()
    assert s["total"] == 2
    assert s["all_ready"] is False
    assert s["any_active"] is True
    assert s["done_count"] == 0

    mw.set_stage("embedder", mw.STAGE_LOADING, "downloading")
    assert mw.snapshot()["any_active"] is True

    mw.set_stage("embedder", mw.STAGE_READY, "ok")
    s = mw.snapshot()
    assert s["done_count"] == 1
    assert s["percent"] == 50.0
    assert s["all_ready"] is False  # STT still pending

    mw.set_stage("stt:parakeet", mw.STAGE_READY)
    s = mw.snapshot()
    assert s["all_ready"] is True
    assert s["any_active"] is False
    assert s["percent"] == 100.0


def test_error_and_skipped_count_as_terminal():
    _reset()
    mw.register("a", "A")
    mw.register("b", "B")
    mw.set_stage("a", mw.STAGE_ERROR, "boom")
    mw.set_stage("b", mw.STAGE_SKIPPED)
    s = mw.snapshot()
    # Both terminal (even though not "ready") → modal can dismiss.
    assert s["all_ready"] is True
    assert s["any_active"] is False


def test_snapshot_shape_for_ui():
    _reset()
    mw.register("embedder", "Language understanding (bge-m3)")
    mw.set_stage("embedder", mw.STAGE_LOADING, "Downloading…")
    s = mw.snapshot()
    m = s["models"][0]
    assert m["key"] == "embedder"
    assert m["name"] == "Language understanding (bge-m3)"
    assert m["stage"] == "loading"
    assert m["detail"] == "Downloading…"


def test_order_preserved():
    _reset()
    for k in ("embedder", "stt:parakeet", "stt:qwen_asr"):
        mw.register(k, k)
    keys = [m["key"] for m in mw.snapshot()["models"]]
    assert keys == ["embedder", "stt:parakeet", "stt:qwen_asr"]
