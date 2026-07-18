"""Request decomposition (agent-orchestration R1).

`decompose(request) -> [SubTask]` splits a multi-goal request into ordered
sub-tasks with explicit dependencies; a single simple goal returns ``[]`` so the
caller runs the existing single agent/answer path (R1.2, Property 1).
Deterministic (cues: enumerations, "and then", multiple distinct imperatives);
never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MAX_SUBTASKS_DEFAULT = 6

# Imperative coding/work verbs that signal a distinct goal.
_VERBS = (
    "review", "analyze", "analyse", "refactor", "implement", "build", "create",
    "write", "add", "fix", "migrate", "design", "document", "test", "optimize",
    "optimise", "summarize", "summarise", "research", "investigate", "generate",
    "deploy", "set up", "setup", "plan",
)
# Connectors that separate sequential goals.
_SPLIT_RE = re.compile(
    r"\b(?:and then|then|after that|next,|finally,|;|\.\s+(?=[A-Z]))",
    re.IGNORECASE)
_NUM_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+)$")


@dataclass
class SubTask:
    id: int
    text: str
    deps: list[int] = field(default_factory=list)


def _cfg_max() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.orchestration, "max_subtasks", 6)))
    except Exception:  # noqa: BLE001
        return _MAX_SUBTASKS_DEFAULT


def decompose(request: str) -> list[SubTask]:
    """Split a multi-goal request into ordered, dependency-linked sub-tasks.
    Single simple goal → []. Never raises."""
    try:
        return _decompose(request)
    except Exception:  # noqa: BLE001
        return []


def _verb_count(text: str) -> int:
    low = text.lower()
    return sum(1 for v in _VERBS if re.search(rf"\b{re.escape(v)}\b", low))


def _decompose(request: str) -> list[SubTask]:
    text = (request or "").strip()
    if not text:
        return []

    # 1) Explicit enumerated/bulleted goals.
    items = []
    for line in text.splitlines():
        m = _NUM_ITEM_RE.match(line)
        if m:
            items.append(m.group(1).strip())

    # 2) Else split on sequential connectors when there are multiple verbs.
    if not items:
        parts = [p.strip(" .,-") for p in _SPLIT_RE.split(text) if p and p.strip(" .,-;")]
        parts = [p for p in parts if len(p) > 3]
        if len(parts) >= 2 and _verb_count(text) >= 2:
            items = parts

    # A single simple goal → no decomposition (R1.2).
    if len(items) < 2:
        return []

    cap = _cfg_max()
    items = items[:cap]
    # Sequential dependency chain by default; the caller's planner may relax to
    # parallel where independent. Each task depends on the previous one unless
    # it reads independent (no back-reference pronoun).
    subs: list[SubTask] = []
    for i, it in enumerate(items):
        deps: list[int] = []
        if i > 0:
            # Independent if it doesn't reference prior work ("it"/"that"/"the").
            low = it.lower()
            references_prior = any(w in low.split() for w in
                                   ("it", "that", "this", "them", "those"))
            # Heuristic: a fresh imperative verb at the start → can run in
            # parallel with the first; otherwise chain to the previous.
            starts_verb = any(low.startswith(v) for v in _VERBS)
            if references_prior or not starts_verb:
                deps = [subs[i - 1].id]
        subs.append(SubTask(id=i, text=it, deps=deps))
    return subs


__all__ = ["decompose", "SubTask"]
