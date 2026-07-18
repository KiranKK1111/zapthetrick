"""Central decision metrics (ArchitectureVerdict Phase 6 — the doc's "policy
quality metrics": measure the clarifier, don't guess).

In-process counters over the two decision surfaces built in Phases 1-4:
  * the pre-gate decision (answer/clarify/defer + which policy rule fired);
  * artifact validation outcomes (validated / repaired / degraded / failed).

Exposed via `GET /api/obs/decisions` and `snapshot()`. Thread-safe, bounded,
fail-open (recording must never break a turn); counters reset on process
restart — durable history can be layered on later without changing callers.
"""
from __future__ import annotations

import threading
import time
from typing import Any

_LOCK = threading.Lock()
_STARTED = time.time()
_GATE: dict[str, int] = {}          # decision -> count
_RULES: dict[str, int] = {}         # policy rule id -> count
_ARTIFACTS = {"validated": 0, "repaired": 0, "degraded": 0, "failed": 0,
              "skipped": 0}


def record_gate_decision(decision: str, rule_id: str | None = None) -> None:
    try:
        with _LOCK:
            d = (decision or "unknown").lower()
            _GATE[d] = _GATE.get(d, 0) + 1
            if rule_id:
                _RULES[rule_id] = _RULES.get(rule_id, 0) + 1
    except Exception:  # noqa: BLE001 — metrics must never break a turn
        pass


def record_artifact_validation(meta: dict | None) -> None:
    try:
        if not isinstance(meta, dict):
            return
        with _LOCK:
            if meta.get("method") == "disabled":
                return
            if not meta.get("validated"):
                _ARTIFACTS["failed"] += 1
            elif meta.get("degraded_from"):
                _ARTIFACTS["degraded"] += 1
            elif meta.get("repaired"):
                _ARTIFACTS["repaired"] += 1
            elif meta.get("method") == "skipped":
                _ARTIFACTS["skipped"] += 1
            else:
                _ARTIFACTS["validated"] += 1
    except Exception:  # noqa: BLE001
        pass


def snapshot() -> dict[str, Any]:
    try:
        with _LOCK:
            gate = dict(_GATE)
            rules = dict(_RULES)
            artifacts = dict(_ARTIFACTS)
        total = sum(gate.values())
        return {
            "since": _STARTED,
            "gate": {
                "counts": gate,
                "total": total,
                "clarify_rate": round(gate.get("clarify", 0) / total, 4)
                if total else 0.0,
                "answer_rate": round(gate.get("answer", 0) / total, 4)
                if total else 0.0,
            },
            "policy_rules": rules,
            "artifacts": artifacts,
        }
    except Exception:  # noqa: BLE001
        return {"gate": {}, "policy_rules": {}, "artifacts": {}}


def reset_for_tests() -> None:
    with _LOCK:
        _GATE.clear()
        _RULES.clear()
        for k in _ARTIFACTS:
            _ARTIFACTS[k] = 0


__all__ = ["record_gate_decision", "record_artifact_validation", "snapshot",
           "reset_for_tests"]
