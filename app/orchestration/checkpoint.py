"""Runtime transaction / checkpoint persistence (roadmap Phase 4 #9/#15).

Wires `orchestration/state.py::save_state`/`load_state` to a per-workspace
sidecar so a long-running goal run can PAUSE and RESUME from pending work — the
DAG executor checkpoints after each completed node, and a re-run loads the state
and skips the nodes already DONE.

Sidecar: `<workspace>/.zapthetrick/orch_state.json`, reusing the same
prefs-blob shape `state.py` persists into (so no new schema). Pure + fail-open —
a missing/corrupt file yields no state, and a write failure is swallowed.
"""
from __future__ import annotations

import json
import os

from app.orchestration.state import AgentState, load_state, save_state

_DIR = ".zapthetrick"
_FILE = "orch_state.json"


def _path(workspace: str) -> str:
    return os.path.join(workspace or ".", _DIR, _FILE)


def _read_prefs(workspace: str) -> dict:
    try:
        with open(_path(workspace), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_checkpoint(workspace: str, state: AgentState) -> bool:
    """Persist `state` into the workspace sidecar (#9). Never raises."""
    try:
        prefs = _read_prefs(workspace)
        save_state(prefs, state)              # wires state.save_state
        d = os.path.join(workspace or ".", _DIR)
        os.makedirs(d, exist_ok=True)
        tmp = _path(workspace) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh)
        os.replace(tmp, _path(workspace))
        return True
    except Exception:  # noqa: BLE001
        return False


def load_checkpoint(workspace: str, scope: str) -> AgentState | None:
    """Restore the persisted state for `scope`, or None (#15). Never raises."""
    try:
        return load_state(_read_prefs(workspace), scope)   # wires state.load_state
    except Exception:  # noqa: BLE001
        return None


def clear_checkpoint(workspace: str, scope: str | None = None) -> None:
    """Drop the whole checkpoint (or just one scope). Never raises."""
    try:
        if scope is None:
            os.remove(_path(workspace))
            return
        prefs = _read_prefs(workspace)
        store = prefs.get("orch_state")
        if isinstance(store, dict) and scope in store:
            store.pop(scope, None)
            with open(_path(workspace), "w", encoding="utf-8") as fh:
                json.dump(prefs, fh)
    except Exception:  # noqa: BLE001
        pass


def scope_for(workspace: str) -> str:
    """A stable per-workspace checkpoint scope."""
    return "workspace:" + os.path.basename(os.path.normpath(workspace or "default"))


__all__ = ["save_checkpoint", "load_checkpoint", "clear_checkpoint", "scope_for"]
