"""Progressive stage protocol schema (vNext §5.3)."""
from __future__ import annotations

from app.response_arch import stage_protocol as SP


def test_stage_event_is_schema_valid():
    ev = SP.stage_event("dsa.verifier", "Running examples", state="active",
                        parent="dsa", progress=0.66, detail="2/3 passed",
                        attempt=2)
    assert SP.validate_stage(ev) == []
    assert ev["id"] == "dsa.verifier"
    assert ev["state"] == "active"
    assert ev["progress"] == 0.66
    assert ev["attempt"] == 2
    assert ev["parent"] == "dsa"
    assert "ts" in ev


def test_unknown_state_falls_back_to_active():
    ev = SP.stage_event("x", state="banana")
    assert ev["state"] == "active"
    assert SP.validate_stage(ev) == []


def test_progress_is_clamped():
    assert SP.stage_event("x", progress=5.0)["progress"] == 1.0
    assert SP.stage_event("x", progress=-1.0)["progress"] == 0.0


def test_label_defaults_to_id():
    assert SP.stage_event("planner")["label"] == "planner"


def test_from_legacy_adapts_the_old_name_shape():
    ev = SP.from_legacy("Running examples")
    assert ev["id"] == "running_examples"       # slug dedups repeats
    assert ev["label"] == "Running examples"
    assert ev["state"] == "active"
    assert SP.validate_stage(ev) == []


def test_validate_flags_missing_id_and_bad_progress():
    assert "id: required" in SP.validate_stage({"state": "active"})
    assert any("progress" in e
               for e in SP.validate_stage({"id": "x", "state": "active",
                                           "progress": 2.0}))
    assert any("state" in e for e in SP.validate_stage({"id": "x",
                                                        "state": "nope"}))
