"""Durable background continuation — jobs survive restart (P6 #23)."""
from __future__ import annotations

import os

from app.response_arch.continuation import DurableJobRegistry


def test_start_update_get(tmp_path):
    p = str(tmp_path / "jobs.json")
    reg = DurableJobRegistry(path=p)
    reg.start("j1", "export", {"fmt": "pdf"})
    assert reg.get("j1")["status"] == "running"
    reg.update("j1", status="done", result={"url": "/x"})
    assert reg.get("j1")["status"] == "done"
    assert reg.get("j1")["result"]["url"] == "/x"
    assert os.path.exists(p)


def test_survives_restart(tmp_path):
    p = str(tmp_path / "jobs.json")
    reg = DurableJobRegistry(path=p)
    reg.start("done-job", "export")
    reg.update("done-job", status="done")
    reg.start("running-job", "research")   # left running

    # Simulate a restart: a fresh registry reads the same file.
    reg2 = DurableJobRegistry(path=p)
    assert reg2.get("done-job")["status"] == "done"
    # A job that was 'running' when we died becomes 'interrupted'.
    r = reg2.get("running-job")
    assert r["status"] == "interrupted" and r["interrupted"] is True
    assert any(j["id"] == "running-job" for j in reg2.pending())


def test_corrupt_store_starts_clean(tmp_path):
    p = tmp_path / "jobs.json"
    p.write_text("not json{", encoding="utf-8")
    reg = DurableJobRegistry(path=str(p))
    assert reg.all() == []


def test_drop(tmp_path):
    reg = DurableJobRegistry(path=str(tmp_path / "jobs.json"))
    reg.start("j", "x")
    reg.drop("j")
    assert reg.get("j") is None
