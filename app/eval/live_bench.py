"""
Live question-detection benchmark harness (Gap 1).

The synthetic scenario suite (tests/test_scenarios_phase_*) checks that features
FIRE; this harness measures the LIVE pipeline's *accuracy* on an annotated
interview corpus — the doc's #1 recommendation ("measure, don't guess"). Each
row is an interviewer utterance with gold labels:

    {"text": "...", "is_question": true|false,
     "questions": ["...", ...],        # gold sub-questions (multi-question)
     "topic": "kafka", "source": "...", "note": "..."}

It runs the SAME deterministic detection the live fast-path uses
(`question_detection.classifier.heuristic_classify` + `live.events.split_questions`)
and reports the metrics the report defines targets for:

    question-detection precision / recall / F1
    false-answer rate      (non-question predicted as a question)  target < 5%
    multi-question recall  (gold sub-questions recovered)

No LLM / DB / network — CI-runnable in ms. `python -m app.eval.live_bench` prints
the report. To benchmark on REAL interviews, transcribe recordings and append
rows to app/eval/data/live_corpus.jsonl (audio-level STT/latency metrics need the
recordings themselves; this transcript-level harness runs today).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

_DEFAULT = pathlib.Path(__file__).parent / "data" / "live_corpus.jsonl"

# Confidence at/above which the fast-path treats a heuristic hit as a question
# (mirrors routes_ws._run_answer's fast_question_path gate).
_Q_CONF = 0.7


def load_corpus(path: str | pathlib.Path | None = None) -> list[dict]:
    p = pathlib.Path(path) if path else _DEFAULT
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _detect(text: str):
    """(is_question, fast_path) — the detector's decision, and whether the
    DETERMINISTIC fast-path handles it (is_question AND confident enough), i.e.
    it skips the slow detection-LLM round-trip. A gold question the detector
    catches but the fast-path doesn't = correct answer, but slower."""
    from app.question_detection.classifier import heuristic_classify
    h = heuristic_classify(text)
    return bool(h.is_question), bool(h.is_question and h.confidence >= _Q_CONF)


def _predict_questions(text: str) -> list[str]:
    try:
        from app.live import events as _ev
        _ctx, q_text = _ev.split_boundary(text, text)
        return list(_ev.split_questions(q_text or text) or [])
    except Exception:  # noqa: BLE001
        return [text]


def run_corpus(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    rows = load_corpus(path)
    tp = fp = tn = fn = 0
    gold_subq = recovered_subq = 0
    gold_q_total = fast_path_hits = 0
    failures: list[dict] = []

    for row in rows:
        text = row.get("text", "")
        gold_q = bool(row.get("is_question"))
        pred_q, fast = _detect(text)
        if pred_q and gold_q:
            tp += 1
        elif pred_q and not gold_q:
            fp += 1
        elif not pred_q and not gold_q:
            tn += 1
        else:
            fn += 1
        if gold_q:
            gold_q_total += 1
            if fast:
                fast_path_hits += 1
        if pred_q != gold_q:
            failures.append({"text": text, "gold": gold_q, "pred": pred_q,
                             "source": row.get("source"), "note": row.get("note")})
        # Multi-question recall: how many gold sub-questions the splitter recovers.
        gold_list = row.get("questions") or ([text] if gold_q else [])
        if gold_q and len(gold_list) > 1:
            preds = _predict_questions(text)
            gold_subq += len(gold_list)
            recovered_subq += min(len(preds), len(gold_list))

    total = len(rows)
    n_pos = tp + fn          # gold questions
    n_neg = fp + tn          # gold non-questions
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 1.0
    recall = round(tp / n_pos, 4) if n_pos else 1.0
    f1 = round(2 * precision * recall / (precision + recall), 4) \
        if (precision + recall) else 0.0
    return {
        "total": total,
        "accuracy": round((tp + tn) / total, 4) if total else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "false-answer rate": non-questions we'd have answered / all non-questions.
        "false_answer_rate": round(fp / n_neg, 4) if n_neg else 0.0,
        "multi_question_recall": round(recovered_subq / gold_subq, 4)
            if gold_subq else 1.0,
        # Latency signal: of the gold questions the detector caught, how many the
        # DETERMINISTIC fast-path handles (skip the slow detection-LLM). The rest
        # are answered CORRECTLY but pay an extra LLM round-trip → a tuning target.
        "fast_path_coverage": round(fast_path_hits / gold_q_total, 4)
            if gold_q_total else 1.0,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "failures": failures,
    }


if __name__ == "__main__":
    import pprint
    report = run_corpus()
    pprint.pprint({k: v for k, v in report.items() if k != "failures"})
    for f in report["failures"]:
        print("FAIL:", f)


__all__ = ["load_corpus", "run_corpus"]
