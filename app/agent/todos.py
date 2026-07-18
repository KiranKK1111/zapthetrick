"""Live task checklist — TodoWrite parity (P2-4, report_2 §P2-4).

Claude Code's most *visible* long-horizon behavior is the live TODO list: it
breaks a big task into a dependency-ordered checklist, keeps exactly one item
`in_progress`, and ticks items off as it goes — streamed to the UI the whole
time. This module models that list and persists it per-workspace so it survives
across the goal loop's fresh-per-round agents.

  Todo            — {content, status, active_form}
  normalize_todos — coerce a model's raw list into validated Todos
  save/load_todos — persist to `.zapthetrick/todos.json`
  todos_summary   — a compact text block injected back into the agent so it
                    always knows its own plan + progress

Field names mirror Claude's tool: `content` (imperative), `status`
(pending|in_progress|completed), `activeForm` (present-continuous label shown
while running). Deterministic + offline; the loop emits the structured `todo`
SSE event.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

_DIR = ".zapthetrick"
_FILE = "todos.json"

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
_STATUSES = {PENDING, IN_PROGRESS, COMPLETED}
_MAX_TODOS = 40


@dataclass
class Todo:
    content: str
    status: str = PENDING
    active_form: str = ""

    def to_dict(self) -> dict:
        return {"content": self.content, "status": self.status,
                "activeForm": self.active_form or self.content}


def _path(workspace: str) -> str:
    return os.path.join(os.path.realpath(workspace), _DIR, _FILE)


def normalize_todos(raw) -> list[Todo]:
    """Coerce a model's raw todo list into validated Todos.

    Accepts a list of dicts ({content/task/text, status, activeForm/active_form})
    or bare strings. Invalid statuses fall back to `pending`. Caps the count and
    ensures AT MOST ONE item is `in_progress` (the first one wins; the rest are
    demoted to pending) — matching Claude's single-active-task invariant."""
    if not isinstance(raw, list):
        return []
    out: list[Todo] = []
    seen_active = False
    for item in raw[:_MAX_TODOS]:
        if isinstance(item, str):
            content, status, active = item.strip(), PENDING, ""
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("task")
                          or item.get("text") or "").strip()
            status = str(item.get("status") or PENDING).strip().lower()
            active = str(item.get("activeForm") or item.get("active_form")
                         or "").strip()
        else:
            continue
        if not content:
            continue
        if status not in _STATUSES:
            status = PENDING
        if status == IN_PROGRESS:
            if seen_active:
                status = PENDING       # only one active task allowed
            else:
                seen_active = True
        out.append(Todo(content=content, status=status, active_form=active))
    return out


def todos_to_dicts(todos: list[Todo]) -> list[dict]:
    return [t.to_dict() for t in todos]


def progress(todos: list[Todo]) -> tuple[int, int]:
    """(completed, total)."""
    return sum(1 for t in todos if t.status == COMPLETED), len(todos)


def save_todos(workspace: str, todos: list[Todo]) -> bool:
    p = _path(workspace)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(todos_to_dicts(todos), f, indent=2)
        return True
    except OSError:
        return False


def load_todos(workspace: str) -> list[Todo]:
    try:
        with open(_path(workspace), encoding="utf-8") as f:
            return normalize_todos(json.load(f))
    except (OSError, ValueError):
        return []


def clear_todos(workspace: str) -> None:
    try:
        os.remove(_path(workspace))
    except OSError:
        pass


def todos_summary(todos: list[Todo], *, max_chars: int = 1200) -> str:
    """A compact checklist preamble for the agent, or '' when empty."""
    if not todos:
        return ""
    done, total = progress(todos)
    marks = {PENDING: "[ ]", IN_PROGRESS: "[~]", COMPLETED: "[x]"}
    lines = [f"TASK CHECKLIST ({done}/{total} done — keep it updated with the "
             "`todo_write` tool; mark exactly ONE item in_progress, tick items "
             "off as you finish them):"]
    for t in todos:
        label = t.active_form if (t.status == IN_PROGRESS and t.active_form) \
            else t.content
        lines.append(f"  {marks.get(t.status, '[ ]')} {label}")
    text = "\n".join(lines)
    return text[:max_chars]


__all__ = [
    "Todo", "PENDING", "IN_PROGRESS", "COMPLETED",
    "normalize_todos", "todos_to_dicts", "progress",
    "save_todos", "load_todos", "clear_todos", "todos_summary",
]
