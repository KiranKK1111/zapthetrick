"""Unit tests for the staged-progress registry (app/documents/progress.py).

Covers the begin/set_stage/update/finish/fail lifecycle for both ops,
percent clamping/monotonicity, the fail-open contract (calls against
unknown ids never raise), and eviction.
"""
from __future__ import annotations

import pytest

from app.documents import progress


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test gets an empty registry (process-global dict)."""
    with progress._LOCK:
        progress._ENTRIES.clear()
    yield
    with progress._LOCK:
        progress._ENTRIES.clear()


# ---- begin / get ---------------------------------------------------------


def test_begin_upload_shape():
    progress.begin("r1", op="upload")
    e = progress.get("r1")
    assert e is not None
    assert e["op"] == "upload"
    assert e["stages"] == progress.UPLOAD_STAGES
    assert e["stage"] == "upload"
    assert e["stage_index"] == 0
    assert e["total_stages"] == len(progress.UPLOAD_STAGES)
    assert e["percent"] == 0.0
    assert e["done"] is False
    assert e["error"] is None
    assert e["counts"] == {}


def test_begin_delete_uses_delete_stages():
    progress.begin("r1", op="delete")
    e = progress.get("r1")
    assert e["stages"] == progress.DELETE_STAGES
    assert e["stage"] == "vectors"


def test_get_unknown_returns_none():
    assert progress.get("nope") is None


def test_get_returns_a_copy():
    progress.begin("r1", op="upload")
    snap = progress.get("r1")
    snap["percent"] = 999.0
    snap["counts"]["hacked"] = 1
    fresh = progress.get("r1")
    assert fresh["percent"] == 0.0
    assert "hacked" not in fresh["counts"]


def test_begin_restarts_entry():
    progress.begin("r1", op="upload")
    progress.finish("r1")
    progress.begin("r1", op="delete")
    e = progress.get("r1")
    assert e["op"] == "delete"
    assert e["done"] is False
    assert e["percent"] == 0.0


# ---- set_stage / update --------------------------------------------------


def test_set_stage_advances_index_and_percent():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "embed", detail="Embedding chunks 0/10",
                       counts={"chunks": 10, "embedded": 0})
    e = progress.get("r1")
    assert e["stage"] == "embed"
    assert e["stage_index"] == progress.UPLOAD_STAGES.index("embed")
    # 3 stages fully done out of 6 => 50%
    assert e["percent"] == pytest.approx(50.0)
    assert e["detail"] == "Embedding chunks 0/10"
    assert e["counts"] == {"chunks": 10, "embedded": 0}


def test_update_moves_percent_within_stage():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "embed")
    progress.update("r1", fraction=0.5, detail="Embedding chunks 5/10",
                    counts={"embedded": 5})
    e = progress.get("r1")
    # (3 + 0.5) / 6 = 58.3%
    assert e["percent"] == pytest.approx(58.3, abs=0.1)
    assert e["detail"] == "Embedding chunks 5/10"
    assert e["counts"]["embedded"] == 5


def test_counts_merge_not_replace():
    progress.begin("r1", op="upload")
    progress.update("r1", counts={"chunks": 10})
    progress.update("r1", counts={"embedded": 4})
    assert progress.get("r1")["counts"] == {"chunks": 10, "embedded": 4}


def test_percent_clamped_to_0_100():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "ready", fraction=5.0)   # over-fraction
    assert progress.get("r1")["percent"] <= 100.0
    progress.begin("r2", op="upload")
    progress.update("r2", fraction=-3.0)              # under-fraction
    assert progress.get("r2")["percent"] == 0.0
    progress.update("r2", fraction=float("nan"))      # NaN is treated as 0
    assert progress.get("r2")["percent"] == 0.0


def test_percent_never_goes_backwards():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "embed")
    progress.update("r1", fraction=0.9)
    high = progress.get("r1")["percent"]
    progress.update("r1", fraction=0.1)               # late/racing update
    assert progress.get("r1")["percent"] >= high
    progress.set_stage("r1", "chunk")                 # stage rewind attempt
    assert progress.get("r1")["stage_index"] >= progress.UPLOAD_STAGES.index("embed")


def test_unknown_stage_name_holds_position():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "embed")
    progress.set_stage("r1", "not-a-stage", detail="odd")
    e = progress.get("r1")
    assert e["stage_index"] == progress.UPLOAD_STAGES.index("embed")
    assert e["detail"] == "odd"


# ---- finish / fail --------------------------------------------------------


def test_finish_lands_on_last_stage_at_100():
    progress.begin("r1", op="upload")
    progress.set_stage("r1", "embed")
    progress.finish("r1", detail="Ready — 12 chunks indexed")
    e = progress.get("r1")
    assert e["done"] is True
    assert e["error"] is None
    assert e["percent"] == 100.0
    assert e["stage"] == progress.UPLOAD_STAGES[-1]
    assert e["stage_index"] == len(progress.UPLOAD_STAGES) - 1
    assert e["detail"] == "Ready — 12 chunks indexed"


def test_fail_sets_error_and_done():
    progress.begin("r1", op="delete")
    progress.set_stage("r1", "chunks")
    progress.fail("r1", "boom")
    e = progress.get("r1")
    assert e["done"] is True
    assert e["error"] == "boom"
    assert e["percent"] < 100.0          # bar freezes where it failed


def test_updates_after_terminal_state_are_ignored():
    progress.begin("r1", op="upload")
    progress.finish("r1")
    progress.set_stage("r1", "chunk", detail="late")
    progress.update("r1", fraction=0.1, detail="later")
    e = progress.get("r1")
    assert e["done"] is True
    assert e["percent"] == 100.0
    assert e["detail"] != "later"


# ---- fail-open contract ----------------------------------------------------


def test_all_mutators_are_noops_on_unknown_id():
    # None of these may raise, and none may create an entry.
    progress.set_stage("ghost", "embed")
    progress.update("ghost", fraction=0.5, detail="x", counts={"a": 1})
    progress.finish("ghost")
    progress.fail("ghost", "err")
    progress.clear("ghost")
    assert progress.get("ghost") is None


def test_clear_removes_entry():
    progress.begin("r1", op="upload")
    progress.clear("r1")
    assert progress.get("r1") is None


def test_eviction_caps_registry_size():
    for i in range(progress._MAX_ENTRIES + 10):
        progress.begin(f"r{i}", op="upload")
    with progress._LOCK:
        assert len(progress._ENTRIES) <= progress._MAX_ENTRIES
    # Newest entries survive; the oldest were evicted.
    assert progress.get(f"r{progress._MAX_ENTRIES + 9}") is not None
    assert progress.get("r0") is None
