"""Task Center job registry — start/update/finish, newest-first snapshot,
bounded eviction, and clear-finished-keeps-running."""
from __future__ import annotations

from app.obs.jobs import JobRegistry


def test_start_update_finish_snapshot():
    r = JobRegistry(max_jobs=5)
    a = r.start("Export PDF", kind="export")
    b = r.start("Answer", kind="chat")
    r.update(a, progress=0.5)
    r.finish(a, ok=True)
    snap = r.snapshot()
    assert snap[0]["id"] == b  # newest first
    ja = next(j for j in snap if j["id"] == a)
    assert ja["status"] == "done" and ja["progress"] == 0.5
    assert ja["finished"] is not None
    assert next(j for j in snap if j["id"] == b)["status"] == "running"


def test_eviction_is_bounded():
    r = JobRegistry(max_jobs=2)
    ids = [r.start(f"j{i}") for i in range(4)]
    snap = r.snapshot()
    assert len(snap) == 2
    kept = {j["id"] for j in snap}
    assert ids[-1] in kept and ids[-2] in kept and ids[0] not in kept


def test_clear_finished_keeps_running():
    r = JobRegistry()
    a = r.start("done one")
    b = r.start("running one")
    r.finish(a, ok=True)
    r.clear_finished()
    assert [j["id"] for j in r.snapshot()] == [b]


def test_error_finish_marks_error():
    r = JobRegistry()
    a = r.start("bad export", kind="export")
    r.finish(a, ok=False, detail="boom")
    j = r.snapshot()[0]
    assert j["status"] == "error" and j["detail"] == "boom"


def test_routes_import():
    import app.api.routes_jobs as m
    assert m.router is not None
