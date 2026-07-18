"""Behavioral benchmark harness (ArchitectureVerdict Phase 6).

Runs the labeled prompt corpus (`app/eval/data/behavior_corpus.jsonl`, seeded
from SeveralFeatures.md's scenario tables) through the DETERMINISTIC decision
path (`intent_pipeline.assess` + the policy engine) and scores orchestration
behavior — the design doc's own targets:

    unnecessary clarification rate  < 5%   (expected proceed, got clarify)
    missed clarification rate       < 2%   (expected clarify, got proceed)

Each corpus row: {"prompt", "expected": proceed|clarify|defer,
"has_artifact"?, "attachment_slots"?, "known_prefs"?, "source", "note"}.
`defer` counts as proceed (the LLM gate answers; no forced interruption) —
matching the doc's binary Clarifier?✅/❌ framing.

No LLM, no DB, no network — CI-runnable in milliseconds. `python -m
app.eval.behavior_bench` prints the report; tests assert the targets.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

_DEFAULT = pathlib.Path(__file__).parent / "data" / "behavior_corpus.jsonl"


def load_corpus(path: str | pathlib.Path | None = None) -> list[dict]:
    p = pathlib.Path(path) if path else _DEFAULT
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _bucket(decision: str) -> str:
    """clarify vs proceed (answer + defer both proceed — no interruption)."""
    return "clarify" if decision == "clarify" else "proceed"


def run_corpus(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    from app.clarify.intent_pipeline import assess

    rows = load_corpus(path)
    failures: list[dict] = []
    n_expected_proceed = n_expected_clarify = 0
    n_unnecessary = n_missed = 0

    for row in rows:
        expected = _bucket(str(row.get("expected", "proceed")).lower()
                           .replace("defer", "proceed"))
        a = assess(
            row.get("prompt", ""),
            row.get("recent", "") or "",
            row.get("known_prefs") or None,
            has_artifact=bool(row.get("has_artifact")),
            attachment_slots=row.get("attachment_slots") or None,
        )
        got = _bucket(a.decision)
        if expected == "proceed":
            n_expected_proceed += 1
            if got == "clarify":
                n_unnecessary += 1
        else:
            n_expected_clarify += 1
            if got == "proceed":
                n_missed += 1
        if got != expected:
            failures.append({
                "prompt": row.get("prompt"), "expected": expected,
                "got": got, "decision": a.decision, "intent": a.intent,
                "missing_required": a.missing_required,
                "source": row.get("source"), "note": row.get("note"),
            })

    total = len(rows)
    correct = total - len(failures)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 1.0,
        "unnecessary_clarification_rate": round(
            n_unnecessary / n_expected_proceed, 4) if n_expected_proceed else 0.0,
        "missed_clarification_rate": round(
            n_missed / n_expected_clarify, 4) if n_expected_clarify else 0.0,
        "failures": failures,
    }


if __name__ == "__main__":
    import pprint
    report = run_corpus()
    pprint.pprint({k: v for k, v in report.items() if k != "failures"})
    for f in report["failures"]:
        print("FAIL:", f)


__all__ = ["load_corpus", "run_corpus"]
