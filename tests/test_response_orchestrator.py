"""ResponseOrchestrator — the coherent named umbrella (P6 #3)."""
from __future__ import annotations

from app.response_arch.content_router import Shape
from app.response_arch.orchestrator import ResponseOrchestrator


def test_plan_frame_before_tokens():
    orch = ResponseOrchestrator("compare A vs B")
    ev, data = orch.plan_frame()
    assert ev == "plan"
    assert data["shape"] == "comparison"
    assert data["sections"]                       # upcoming sections enumerated
    assert data["stream"]["block_aware"] is True  # shape drives stream mode


def test_prose_turn_emits_no_block_frames():
    orch = ResponseOrchestrator("tell me about the sea")
    assert orch.stream_mode.name == "token"
    frames = orch.on_token("The sea is vast.\n\nIt is deep.\n\n")
    assert frames == []                           # token mode → no block frames


def test_code_turn_streams_blocks_and_artifacts():
    orch = ResponseOrchestrator("implement quicksort in python")
    assert orch.stream_mode.emit_artifacts is True
    frames = []
    text = ("Here is the code.\n\n"
            "```python name=quick.py\ndef q(x):\n    return x\n```\n\nDone.")
    for i in range(0, len(text), 5):
        frames += orch.on_token(text[i:i + 5])
    frames += orch.flush()
    events = [e for e, _ in frames]
    assert "block" in events
    assert "artifact" in events                   # progressive delivery
    art = [d for e, d in frames if e == "artifact"][0]
    assert art["filename"] == "quick.py" and art["progressive"] is True


def test_analytics_frame_has_ttfmu():
    orch = ResponseOrchestrator("hi")
    orch.on_token("hello")
    ev, data = orch.analytics_frame()
    assert ev == "analytics"
    assert "ttfmu_ms" in data and "total_ms" in data


def test_finalize_uses_planned_shape():
    orch = ResponseOrchestrator("compare A vs B")
    shaped = orch.finalize("A is fast. B is slow.")
    assert shaped.shape == Shape.COMPARISON


def test_envelope_carries_shape():
    orch = ResponseOrchestrator("implement quicksort in python")
    env = orch.envelope(model="m")
    assert env["answer"]["shape"] == "code"
