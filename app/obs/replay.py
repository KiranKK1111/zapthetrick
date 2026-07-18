"""Deterministic replay / flight-recorder core (roadmap Phase 1 #4).

Record a component's (inputs -> output), persist it, then REPLAY the same inputs
through the current code and diff against the recorded output. Any drift is a
regression the moment behavior changes — the "record production, replay offline"
capability the roadmap calls a deterministic replay lab.

Complements `obs/trace.py` (which records *what a turn did*): this records
*input->output pairs* so they can be re-executed and compared. Pure + fail-open.

    store = ReplayStore()
    store.add(record("intent", {"text": "hi"}, classify("hi")))
    report = store.replay_all(lambda inp: classify(inp["text"]))
    assert report.all_matched          # green until behavior drifts
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# A handler re-executes a recording's inputs and returns the output to compare.
Handler = Callable[[dict], Any]


def _canonical(value: Any) -> str:
    """Stable JSON for deep-equality comparison (order-independent for dicts)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class Recording:
    kind: str
    inputs: dict
    output: Any
    meta: dict = field(default_factory=dict)
    id: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "inputs": self.inputs,
                "output": self.output, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "Recording":
        return Recording(
            kind=d.get("kind", ""), inputs=d.get("inputs", {}),
            output=d.get("output"), meta=d.get("meta", {}), id=d.get("id", ""),
        )


def record(kind: str, inputs: dict, output: Any, *, meta: dict | None = None,
           id: str = "") -> Recording:
    return Recording(kind=kind, inputs=dict(inputs), output=output,
                     meta=dict(meta or {}), id=id)


@dataclass
class ReplayMismatch:
    id: str
    kind: str
    inputs: dict
    recorded: Any
    actual: Any


@dataclass
class ReplayReport:
    total: int
    matched: int
    mismatches: list[ReplayMismatch] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return self.total > 0 and self.matched == self.total and not self.errors

    @property
    def match_rate(self) -> float:
        return (self.matched / self.total) if self.total else 0.0


def replay_one(rec: Recording, handler: Handler) -> tuple[bool, Any]:
    """Re-run one recording; return (matched, actual_output)."""
    actual = handler(rec.inputs)
    return _canonical(actual) == _canonical(rec.output), actual


class ReplayStore:
    """A collection of recordings, replayable as a batch and JSONL-persistable."""

    def __init__(self) -> None:
        self._recs: list[Recording] = []

    def add(self, rec: Recording) -> None:
        self._recs.append(rec)

    def extend(self, recs: Iterable[Recording]) -> None:
        self._recs.extend(recs)

    def __len__(self) -> int:
        return len(self._recs)

    def to_jsonl(self) -> str:
        return "\n".join(_canonical(r.to_dict()) for r in self._recs)

    @staticmethod
    def from_jsonl(text: str) -> "ReplayStore":
        store = ReplayStore()
        for line in (text or "").splitlines():
            line = line.strip()
            if line:
                store.add(Recording.from_dict(json.loads(line)))
        return store

    def replay_all(self, handler: Handler) -> ReplayReport:
        matched = 0
        mismatches: list[ReplayMismatch] = []
        errors: list[str] = []
        for r in self._recs:
            try:
                ok, actual = replay_one(r, handler)
            except Exception as exc:  # noqa: BLE001 — one bad replay ≠ crash
                errors.append(f"{r.id or r.kind}: {type(exc).__name__}: {exc}")
                continue
            if ok:
                matched += 1
            else:
                mismatches.append(ReplayMismatch(
                    id=r.id or r.kind, kind=r.kind, inputs=r.inputs,
                    recorded=r.output, actual=actual))
        return ReplayReport(total=len(self._recs), matched=matched,
                            mismatches=mismatches, errors=errors)


# ── Runtime flight recorder (Phase 1 #4 — the wired half) ──────────────────
# A process-global recorder that real runtime paths feed via `capture(...)`, so
# the record → replay → diff loop runs against genuine production I/O instead of
# only hand-built fixtures. Bounded, gated, and fail-open: a recorder error can
# never touch the turn it observes. The first wired producer is `obs/trace.py`
# (`build_trace` is called per-turn from the agent runtime), so live traffic
# accumulates replayable (inputs → output) recordings automatically.

_REC_LOCK = threading.Lock()
_RECORDER = ReplayStore()
_REC_CAP = 500


def flight_recorder_enabled() -> bool:
    """Gate for the runtime capture hook. Enabling default: absent config → ON.
    Config: `obs.flight_recorder` (bool)."""
    try:
        from app.core.config_loader import cfg
        sec = getattr(cfg, "obs", None)
        if sec is None:
            return True
        return bool(getattr(sec, "flight_recorder", True))
    except Exception:  # noqa: BLE001
        return True


def capture(kind: str, inputs: dict, output: Any, *, meta: dict | None = None,
            cap: int | None = None) -> bool:
    """Record one (inputs → output) pair from a live runtime path. Gated,
    bounded (ring), and fail-open — returns True iff it actually recorded."""
    try:
        if not flight_recorder_enabled():
            return False
        limit = _REC_CAP if cap is None else int(cap)
        with _REC_LOCK:
            _RECORDER.add(record(kind, inputs, output, meta=meta or {}))
            overflow = len(_RECORDER._recs) - limit
            if overflow > 0:
                del _RECORDER._recs[0:overflow]
        return True
    except Exception:  # noqa: BLE001 — the observed path must never break
        return False


def recorder() -> ReplayStore:
    return _RECORDER


def captured_count() -> int:
    with _REC_LOCK:
        return len(_RECORDER)


def replay_captured(handler: Handler) -> ReplayReport:
    """Re-run every captured recording through `handler` and diff. This is the
    end-to-end record → replay → diff the roadmap asks for, over live traffic."""
    with _REC_LOCK:
        snap = ReplayStore()
        snap.extend(list(_RECORDER._recs))
    return snap.replay_all(handler)


def reset_recorder() -> None:
    with _REC_LOCK:
        _RECORDER._recs.clear()


__all__ = [
    "Handler", "Recording", "record", "ReplayMismatch", "ReplayReport",
    "replay_one", "ReplayStore",
    "flight_recorder_enabled", "capture", "recorder", "captured_count",
    "replay_captured", "reset_recorder",
]
