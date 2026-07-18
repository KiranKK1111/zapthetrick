"""Online eval sampling (Architecture §14 / G12.1)."""
from __future__ import annotations

from app.eval.online_sample import maybe_record


def test_records_when_draw_below_rate():
    got: list[dict] = []
    ok = maybe_record(question="q", answer="a", intent="knowledge",
                      trace_id="t1", rate=0.5, rng=lambda: 0.1, sink=got.append)
    assert ok is True
    assert got[0]["kind"] == "eval_sample"
    assert got[0]["question"] == "q" and got[0]["trace_id"] == "t1"
    assert got[0]["intent"] == "knowledge"


def test_skips_when_draw_above_rate():
    got: list[dict] = []
    ok = maybe_record(question="q", answer="a", rate=0.5, rng=lambda: 0.9,
                      sink=got.append)
    assert ok is False and got == []


def test_rate_zero_never_records():
    got: list[dict] = []
    assert maybe_record(question="q", answer="a", rate=0.0, rng=lambda: 0.0,
                        sink=got.append) is False
    assert got == []


def test_truncates_long_fields():
    got: list[dict] = []
    maybe_record(question="x" * 5000, answer="y" * 9000, rate=1.0,
                 rng=lambda: 0.0, sink=got.append)
    assert len(got[0]["question"]) == 2000
    assert len(got[0]["answer"]) == 4000


def test_fail_open_on_bad_sink():
    def boom(_rec):
        raise RuntimeError("sink down")
    assert maybe_record(question="q", answer="a", rate=1.0, rng=lambda: 0.0,
                        sink=boom) is False
