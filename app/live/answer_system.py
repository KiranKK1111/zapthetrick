"""Technical answer system — role × class × envelope × depth (vNext §4.13, Stage 7 H).

A great interview answer is shaped by four things at once: the ROLE lens (a
backend answer weights data modeling + failure handling; a frontend answer
weights UX + rendering), the question CLASS, the resume ENVELOPE (you may only
claim what you've genuinely done — anything beyond it is honest-framed, never
fabricated), and the DEPTH ladder (a drill-down goes DEEPER, it never restarts at
the definition). §4.13 assembles these; this module owns the deterministic parts:

  * **role lens** — detect the interview's role from the title + JD + skills and
    emit the angle to emphasize;
  * **depth ladder L1–L4** — per-session, per-topic depth that only ever ADVANCES
    on a drill-down ("go deeper"), so follow-ups build instead of repeating;
  * **envelope honest-framing** — an out-of-envelope claim (not in the Career
    Graph's grounded facts, Stage-6 J) triggers an honest-frame directive;
  * **unknown-question strategy** — a first-principles scaffold.

The knowledge snippets (`knowledge.py`), qtype strategy (`strategy.py`), and depth
estimate (`objective.py`) already exist and feed this. Pure + fail-open. Flag-
gated (`live.answer_system`, default OFF → today's generic shaping).
"""
from __future__ import annotations

import re
import threading

_LOCK = threading.RLock()


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "answer_system", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Role lens
# --------------------------------------------------------------------------- #
_ROLE_CUES: dict[str, tuple[str, ...]] = {
    "backend": ("backend", "api", "server", "microservice", "database", "sql",
                "grpc", "queue", "kafka", "distributed", "throughput"),
    "frontend": ("frontend", "react", "vue", "angular", "css", "ui", "ux",
                 "browser", "typescript", "accessibility", "rendering"),
    "sre": ("sre", "devops", "kubernetes", "terraform", "monitoring", "incident",
            "on-call", "oncall", "reliability", "observability", "prometheus"),
    "data": ("data engineer", "etl", "pipeline", "warehouse", "spark", "airflow",
             "dbt", "batch", "streaming", "lakehouse"),
    "ml": ("machine learning", "ml", "model", "training", "pytorch", "tensorflow",
           "inference", "embedding", "llm", "fine-tune", "feature"),
    "security": ("security", "appsec", "pentest", "vulnerability", "crypto",
                 "auth", "oauth", "threat", "exploit", "hardening"),
    "qa": ("qa", "quality assurance", "test automation", "selenium", "cypress",
           "regression", "test plan"),
    "mobile": ("mobile", "android", "ios", "flutter", "swift", "kotlin",
               "react native"),
}
_ROLE_ANGLE: dict[str, str] = {
    "backend": "correctness, data modeling, API design, and failure handling",
    "frontend": "user experience, rendering performance, accessibility, and state management",
    "sre": "reliability, observability, incident response, and blast-radius control",
    "data": "data correctness, pipeline reliability, schema evolution, and scale",
    "ml": "problem framing, data quality, evaluation, and inference cost",
    "security": "threat modeling, least privilege, defense in depth, and honest risk framing",
    "qa": "coverage, edge cases, reproducibility, and test design",
    "mobile": "UX, offline/limited connectivity, battery/memory, and platform constraints",
    "generalist": "clear reasoning, tradeoffs, and a concrete outcome",
}


def detect_role(role: str = "", jd_text: str = "", skills=None) -> str:
    """Best role lens from the title + JD + skills (cue overlap). 'generalist'
    when nothing scores. Never raises."""
    try:
        blob = " ".join([str(role or ""), str(jd_text or ""),
                         " ".join(str(s) for s in (skills or []))]).lower()
        if not blob.strip():
            return "generalist"
        best, best_n = "generalist", 0
        for r, cues in _ROLE_CUES.items():
            n = sum(1 for c in cues if c in blob)
            if n > best_n:
                best, best_n = r, n
        return best
    except Exception:  # noqa: BLE001
        return "generalist"


def role_directive(role: str) -> str:
    angle = _ROLE_ANGLE.get((role or "").strip().lower(),
                            _ROLE_ANGLE["generalist"])
    return f"Answer through a {role or 'generalist'} lens — weight {angle}."


# --------------------------------------------------------------------------- #
# Depth ladder L1–L4 (progresses on drill-down, never restarts)
# --------------------------------------------------------------------------- #
L1, L2, L3, L4 = 1, 2, 3, 4
_DEPTH_LABEL = {
    L1: "headline + a crisp definition",
    L2: "the mechanism — how it actually works",
    L3: "tradeoffs and internals",
    L4: "edge cases, failure modes, and source-level detail",
}
_ladder: dict[str, int] = {}          # f"{session}:{topic}" -> level


def _key(session_id: str, topic: str) -> str:
    return f"{session_id or ''}:{(topic or '').strip().lower()}"


def depth(session_id: str, topic: str) -> int:
    with _LOCK:
        return _ladder.get(_key(session_id, topic), L1)


def advance(session_id: str, topic: str) -> int:
    """A drill-down ("go deeper") ADVANCES the depth for this topic, capped at
    L4; it never restarts. Returns the new level."""
    with _LOCK:
        k = _key(session_id, topic)
        lvl = min(L4, _ladder.get(k, L1) + 1)
        _ladder[k] = lvl
        return lvl


def set_depth(session_id: str, topic: str, level: int) -> None:
    with _LOCK:
        _ladder[_key(session_id, topic)] = max(L1, min(L4, int(level)))


def depth_directive(level: int) -> str:
    lbl = _DEPTH_LABEL.get(max(L1, min(L4, int(level or L1))), _DEPTH_LABEL[L1])
    return (f"Pitch this at depth L{max(L1, min(L4, int(level or L1)))}: {lbl}. "
            "If this is a drill-down, go DEEPER — do not restart from the "
            "definition.")


def forget_session(session_id: str) -> None:
    pref = f"{session_id or ''}:"
    with _LOCK:
        for k in [k for k in _ladder if k.startswith(pref)]:
            _ladder.pop(k, None)


def reset_for_tests() -> None:
    with _LOCK:
        _ladder.clear()


# --------------------------------------------------------------------------- #
# Envelope honest-framing + unknown strategy
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9+#.]{3,}")


def in_envelope(claim: str, grounded_terms) -> bool:
    """Whether the claim's key terms are supported by the resume envelope (the
    Career Graph's grounded facts, Stage-6 J). Empty envelope → treat as NOT in
    envelope (honest-frame) so we never fabricate on a missing resume."""
    try:
        blob = " ".join(str(t) for t in (grounded_terms or [])).lower()
        if not blob:
            return False
        terms = set(_WORD.findall((claim or "").lower()))
        if not terms:
            return True
        hit = sum(1 for t in terms if t in blob)
        return hit / len(terms) >= 0.5
    except Exception:  # noqa: BLE001
        return True


def honest_frame_directive() -> str:
    """§4.13 — an out-of-envelope claim must be honest-framed, never fabricated."""
    return ("This goes beyond what your resume demonstrates. Answer HONESTLY: "
            "say what you HAVE done that's adjacent, then how you would approach "
            "the rest — never claim direct experience you don't have.")


def unknown_directive() -> str:
    """§4.13 — the strategy for a question you don't know: reason from first "
    "principles, out loud, structured."""
    return ("You don't have this memorized — reason from FIRST PRINCIPLES out "
            "loud: state what you DO know, define the problem, work through it "
            "step by step, and flag your assumptions. A structured attempt beats "
            "a confident guess.")


__all__ = ["enabled", "detect_role", "role_directive", "L1", "L2", "L3", "L4",
           "depth", "advance", "set_depth", "depth_directive", "in_envelope",
           "honest_frame_directive", "unknown_directive", "forget_session",
           "reset_for_tests"]
