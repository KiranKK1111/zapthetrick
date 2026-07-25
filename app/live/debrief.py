"""Assisted-session debrief (vNext §4.17, Stage 11 Component D).

When an assisted Live session ends, a debrief is auto-generated in the background
(rendered through the §3.12 doc loop): a **delivery map** (what was actually said
vs shown — §4.14), the **true claims ledger** (the said-state — §4.8), a
**situation replay** (§4.15), **panel dynamics** (§4.16), and the likely
**next-round follow-ups** (one-tap → a Coach session, Component C).

Three hard rules:
  * **descriptive, never scored** — it maps what happened; it does NOT grade the
    candidate or predict the hiring outcome;
  * **comp excluded** — any compensation / salary content is stripped (§11.3);
  * **private by default** — the debrief is the candidate's own, not shared.

This module owns the deterministic assembly from duck-typed session data; the doc
render is the §3.12 seam. Pure + fail-open. Flag-gated (`live.debrief`, default
OFF → no debrief).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "debrief", False))
    except Exception:  # noqa: BLE001
        return False


# Compensation / salary content is excluded from the debrief (§11.3).
_COMP_RE = re.compile(
    r"\b(salary|compensation|ctc|comp\b|pay\b|package|lpa|\$\d|₹\d|"
    r"stock|equity|bonus|offer amount|base pay|total comp)\b", re.IGNORECASE)


def _is_comp(text: str) -> bool:
    try:
        return bool(_COMP_RE.search(text or ""))
    except Exception:  # noqa: BLE001
        return False


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
# Section builders (each pure, comp-excluded)
# --------------------------------------------------------------------------- #
def _delivery_map(deliveries) -> list:
    """From §4.14 delivery reads: per-question delivered ratio + whether it was
    interrupted + any improvised additions. Comp-excluded."""
    out = []
    for d in deliveries or ():
        q = str(_get(d, "question", "") or "")
        if _is_comp(q):
            continue
        ratio = float(_get(d, "delivered_ratio", 0.0) or 0.0)
        out.append({"question": q[:200],
                    "delivered_ratio": round(ratio, 3),
                    "completed": bool(_get(d, "completed", ratio >= 0.9)),
                    "improvised": len(_get(d, "improvised", []) or [])})
    return out


def _claims_ledger(claims) -> list:
    """The §4.8 said-state — the claims the candidate actually made. Comp lines
    dropped; deduped, order preserved."""
    out, seen = [], set()
    for c in claims or ():
        text = (str(c).strip() if not isinstance(c, dict) else str(c.get("text", "")).strip())
        if not text or _is_comp(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text[:300])
    return out


def _situation_replay(situations) -> list:
    """The §4.15 situation sequence (stress/harshness/conviction-trap/rapport…),
    salary situations excluded."""
    out = []
    for s in situations or ():
        name = str(_get(s, "situation", s) if not isinstance(s, str) else s)
        if name in ("salary",) or _is_comp(name):
            continue
        out.append({"situation": name,
                    "confidence": round(float(_get(s, "confidence", 0.0) or 0.0), 3)}
                   if not isinstance(s, str) else {"situation": name})
    return out


def _panel_dynamics(panel) -> list:
    """The §4.16 panel roster (P1/P2/P3 + inferred role), attribution counts."""
    out = []
    for p in panel or ():
        out.append({"id": str(_get(p, "id", "") or ""),
                    "role": str(_get(p, "role", "") or ""),
                    "turns": int(_get(p, "turns", 0) or 0)})
    return out


def _follow_ups(topics, *, predict_fn=None, max_n: int = 5) -> list:
    """Likely next-round follow-ups (one-tap → a Coach session). Uses an INJECTED
    `predict_fn(topics) -> [questions]` (reuses `live.predict`), else a light
    template over the covered topics. Comp topics excluded."""
    try:
        clean = [t for t in (topics or []) if t and not _is_comp(str(t))]
        if predict_fn is not None:
            try:
                qs = predict_fn(clean) or []
                return [str(q) for q in qs if q and not _is_comp(str(q))][:max_n]
            except Exception:  # noqa: BLE001
                pass
        return [f"Go deeper on {t}: what would you change at scale?"
                for t in clean[:max_n]]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass
class SessionDebrief:
    delivery_map: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    situations: list = field(default_factory=list)
    panel: list = field(default_factory=list)
    follow_ups: list = field(default_factory=list)
    private: bool = True           # private-by-default (§4.17)
    scored: bool = False           # DESCRIPTIVE — never a grade

    def to_dict(self) -> dict:
        return {"delivery_map": self.delivery_map, "claims": self.claims,
                "situations": self.situations, "panel": self.panel,
                "follow_ups": self.follow_ups, "private": self.private,
                "scored": self.scored,
                "disclaimer": "Descriptive session map — not a score or a "
                              "hiring prediction. Private to you."}


def build_debrief(*, deliveries=None, claims=None, situations=None, panel=None,
                  topics=None, predict_fn=None) -> SessionDebrief:
    """Assemble the descriptive, comp-excluded, private debrief from a session's
    Live data (all duck-typed, all optional). Disabled → an empty (still private)
    debrief. Never raises."""
    try:
        if not enabled():
            return SessionDebrief()
        return SessionDebrief(
            delivery_map=_delivery_map(deliveries),
            claims=_claims_ledger(claims),
            situations=_situation_replay(situations),
            panel=_panel_dynamics(panel),
            follow_ups=_follow_ups(topics, predict_fn=predict_fn),
            private=True, scored=False)
    except Exception:  # noqa: BLE001
        return SessionDebrief()


def to_markdown(debrief: SessionDebrief) -> str:
    """Render the debrief as Markdown for the §3.12 doc loop. Descriptive
    headings; no scores. Never raises → ''."""
    try:
        out = ["# Session debrief", "", "*" + debrief.to_dict()["disclaimer"] + "*", ""]
        if debrief.delivery_map:
            out.append("## Delivery")
            for d in debrief.delivery_map:
                pct = int(round(d["delivered_ratio"] * 100))
                tail = "" if d["completed"] else " (interrupted)"
                imp = f" · {d['improvised']} improvised" if d["improvised"] else ""
                out.append(f"- {d['question']} — delivered {pct}%{tail}{imp}")
            out.append("")
        if debrief.claims:
            out.append("## What you said")
            out += [f"- {c}" for c in debrief.claims]
            out.append("")
        if debrief.situations:
            out.append("## Situation replay")
            out += [f"- {s['situation']}" for s in debrief.situations]
            out.append("")
        if debrief.panel:
            out.append("## Panel")
            out += [f"- {p['id']} ({p['role'] or 'interviewer'}) — {p['turns']} turns"
                    for p in debrief.panel]
            out.append("")
        if debrief.follow_ups:
            out.append("## Practice these follow-ups")
            out += [f"- {q}" for q in debrief.follow_ups]
        return "\n".join(out).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["enabled", "SessionDebrief", "build_debrief", "to_markdown"]
