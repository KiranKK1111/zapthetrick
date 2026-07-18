"""Accuracy ledger — the live pipeline's decision log + user feedback.

Every utterance decision (answered / skipped / promoted / forced) is recorded
with its reason and detector signals, and the user can flag any decision as
wrong from the UI ("should have answered" / "shouldn't have answered"). This
turns every real interview into labeled data: the summary shows where the
detectors miss, and the JSONL file is the training/tuning corpus.

In-process counters (cheap, per run) + append-only JSONL persistence
(``data/live_ledger.jsonl``). Everything is fail-open: a ledger error must
never affect the live session.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter

from app.core.config_loader import cfg

log = logging.getLogger("zapthetrick.ledger")

# Decision kinds.
ANSWERED = "answered"
SKIPPED = "skipped"
PROMOTED = "promoted"   # answered via the promotion layer (indirect/tonal)
FORCED = "forced"       # user tapped "Answer" on a skipped utterance

_lock = threading.Lock()
_counts: Counter = Counter()
_feedback_counts: Counter = Counter()
# Per-session counters (enhancement #3, 2026-07-08): session health can show
# e.g. how many duplicate questions were suppressed in THIS interview.
_session_counts: dict[str, Counter] = {}


def _enabled() -> bool:
    return bool(getattr(cfg.live, "accuracy_ledger", True))


def _path() -> str:
    p = str(getattr(cfg.live, "ledger_path", "") or "data/live_ledger.jsonl")
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return p


def _append(entry: dict) -> None:
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock:
            with open(_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — never break the live session
        log.debug("ledger append failed: %s", exc)


def record(sid: str, qid: str | None, utterance: str, decision: str,
           *, reason: str = "", qtype: str = "", signals: dict | None = None,
           ) -> None:
    """Record one utterance decision. Never raises."""
    if not _enabled():
        return
    try:
        with _lock:
            _counts[decision] += 1
            if reason:
                _counts[f"{decision}:{reason}"] += 1
            sc = _session_counts.setdefault(sid, Counter())
            sc[decision] += 1
            if reason:
                sc[f"{decision}:{reason}"] += 1
            while len(_session_counts) > 64:      # bounded FIFO
                _session_counts.pop(next(iter(_session_counts)))
        _append({
            "kind": "decision",
            "ts": time.time(),
            "sid": sid,
            "qid": qid,
            "utterance": (utterance or "")[:300],
            "decision": decision,
            "reason": reason,
            "qtype": qtype,
            "signals": signals or {},
        })
    except Exception:  # noqa: BLE001
        pass


def session_counts(sid: str) -> dict:
    """This session's decision counters (answered / skipped:reason / ...) —
    consumed by session health (duplicate-suppression visibility)."""
    try:
        with _lock:
            return dict(_session_counts.get(sid) or {})
    except Exception:  # noqa: BLE001
        return {}


def feedback(sid: str, qid: str | None, verdict: str,
             *, utterance: str = "") -> None:
    """Record a user's correction of a decision. `verdict` is one of
    'should_have_answered' | 'should_not_have_answered'. Never raises."""
    if not _enabled():
        return
    try:
        with _lock:
            _feedback_counts[verdict] += 1
        _append({
            "kind": "feedback",
            "ts": time.time(),
            "sid": sid,
            "qid": qid,
            "verdict": verdict,
            "utterance": (utterance or "")[:300],
        })
    except Exception:  # noqa: BLE001
        pass


def answer_feedback(sid: str, qid: str | None, rating: str,
                    *, utterance: str = "", answer: str = "") -> None:
    """Record a thumbs up/down on a delivered live answer. `rating` is one of
    'thumb_up' | 'thumb_down' | '' (cleared). Never raises."""
    if not _enabled():
        return
    try:
        with _lock:
            if rating:
                _feedback_counts[f"answer:{rating}"] += 1
        _append({
            "kind": "answer_feedback",
            "ts": time.time(),
            "sid": sid,
            "qid": qid,
            "rating": rating,
            "utterance": (utterance or "")[:300],
            "answer": (answer or "")[:500],
        })
    except Exception:  # noqa: BLE001
        pass


_history_loaded = False


def _load_feedback_history() -> None:
    """Seed the in-process feedback counters from the JSONL tail once per
    process, so the learned bias survives restarts. Bounded read; fail-open."""
    global _history_loaded
    if _history_loaded:
        return
    _history_loaded = True
    try:
        p = _path()
        if not os.path.isfile(p):
            return
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 512 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
        with _lock:
            for line in tail.splitlines()[-2000:]:
                try:
                    e = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if e.get("kind") == "feedback" and e.get("verdict"):
                    _feedback_counts[str(e["verdict"])] += 1
    except Exception:  # noqa: BLE001
        pass


def answer_bias() -> float:
    """Self-tuning (2026-07-09): a small detection-threshold bias learned
    from the user's own corrections. "Should have answered" flags push the
    live pipeline toward answering MORE (positive bias lowers thresholds);
    "shouldn't have answered" flags push it toward answering LESS. Bounded
    to ±0.10 and inert until at least 3 corrections exist — the static
    defaults stay authoritative for fresh installs. Never raises."""
    try:
        _load_feedback_history()
        with _lock:
            more = _feedback_counts.get("should_have_answered", 0)
            less = _feedback_counts.get("should_not_have_answered", 0)
        total = more + less
        if total < 3:
            return 0.0
        return max(-0.10, min(0.10, 0.02 * (more - less)))
    except Exception:  # noqa: BLE001
        return 0.0


def summary() -> dict:
    """In-process counters since startup (cheap; the JSONL is the full log)."""
    with _lock:
        return {
            "decisions": dict(_counts),
            "feedback": dict(_feedback_counts),
        }


def reset_for_tests() -> None:
    global _history_loaded
    with _lock:
        _counts.clear()
        _feedback_counts.clear()
        _session_counts.clear()
    # Tests must not absorb the real on-disk feedback history.
    _history_loaded = True
