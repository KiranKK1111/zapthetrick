"""
Cross-round interview memory graph (roadmap Phase 2 #24 / 2C-24).

Per-session memory (world_model / topic_graph) forgets everything the moment a
session ends. A candidate often interviews with the SAME company across multiple
rounds; this keeps a tiny, durable link graph — (company/role) → topics already
covered, qtypes seen, and how many rounds — persisted to a JSON file so a later
round can avoid re-treading ground and surface "you covered X last round".

Deliberately minimal + local: a single JSON file (no schema/DB migration),
namespaced by a normalized (company, role) key. Advisory + fail-open: any I/O
error degrades to today's per-session-only behavior.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field

_LOCK = threading.RLock()


def _default_path() -> pathlib.Path:
    base = os.environ.get("ZAPTHETRICK_DATA_DIR")
    root = pathlib.Path(base) if base else (pathlib.Path.home() / ".zapthetrick")
    return root / "cross_round.json"


def _key(company: str, role: str = "") -> str:
    c = (company or "").strip().lower()
    r = (role or "").strip().lower()
    if not c and not r:
        return ""
    return f"{c}::{r}"


@dataclass
class RoundLink:
    rounds: int = 0
    topics: dict = field(default_factory=dict)   # topic -> count
    qtypes: dict = field(default_factory=dict)   # qtype -> count
    last_ts: float = 0.0

    def to_dict(self) -> dict:
        return {"rounds": self.rounds, "topics": self.topics,
                "qtypes": self.qtypes, "last_ts": self.last_ts}

    @staticmethod
    def from_dict(d: dict) -> "RoundLink":
        try:
            return RoundLink(rounds=int(d.get("rounds", 0)),
                             topics=dict(d.get("topics", {})),
                             qtypes=dict(d.get("qtypes", {})),
                             last_ts=float(d.get("last_ts", 0.0)))
        except Exception:  # noqa: BLE001
            return RoundLink()


def _load(path: pathlib.Path | None) -> dict:
    p = path or _default_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save(data: dict, path: pathlib.Path | None) -> None:
    p = path or _default_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001
        pass


def record_topic(company: str, topic: str, *, role: str = "",
                 qtype: str = "", path: pathlib.Path | None = None) -> None:
    """Record that `topic` (and its qtype) was covered for (company, role).
    Never raises."""
    try:
        k = _key(company, role)
        t = (topic or "").strip().lower()
        if not k or not t:
            return
        with _LOCK:
            data = _load(path)
            link = RoundLink.from_dict(data.get(k, {}))
            link.topics[t] = int(link.topics.get(t, 0)) + 1
            if qtype:
                link.qtypes[qtype] = int(link.qtypes.get(qtype, 0)) + 1
            link.last_ts = time.time()
            data[k] = link.to_dict()
            _save(data, path)
    except Exception:  # noqa: BLE001
        pass


def start_round(company: str, *, role: str = "",
                path: pathlib.Path | None = None) -> int:
    """Mark a new interview round for (company, role); returns the round number
    (1-based). Never raises → 0."""
    try:
        k = _key(company, role)
        if not k:
            return 0
        with _LOCK:
            data = _load(path)
            link = RoundLink.from_dict(data.get(k, {}))
            link.rounds += 1
            link.last_ts = time.time()
            data[k] = link.to_dict()
            _save(data, path)
            return link.rounds
    except Exception:  # noqa: BLE001
        return 0


def prior_link(company: str, *, role: str = "",
               path: pathlib.Path | None = None) -> RoundLink | None:
    """The persisted link for (company, role), or None. Never raises."""
    try:
        k = _key(company, role)
        if not k:
            return None
        data = _load(path)
        if k not in data:
            return None
        return RoundLink.from_dict(data[k])
    except Exception:  # noqa: BLE001
        return None


def prior_topics(company: str, *, role: str = "", top: int = 8,
                 path: pathlib.Path | None = None) -> list[str]:
    """Topics already covered in prior rounds, most-covered first. Never raises."""
    try:
        link = prior_link(company, role=role, path=path)
        if link is None or not link.topics:
            return []
        return [t for t, _ in sorted(link.topics.items(),
                                     key=lambda kv: kv[1], reverse=True)][:top]
    except Exception:  # noqa: BLE001
        return []


def link_directive(company: str, current_topic: str = "", *, role: str = "",
                   path: pathlib.Path | None = None) -> str:
    """When the current topic was already covered in a prior round, nudge the
    answer to build on it rather than repeat. '' otherwise. Never raises."""
    try:
        link = prior_link(company, role=role, path=path)
        if link is None or link.rounds < 1:
            return ""
        t = (current_topic or "").strip().lower()
        if t and t in link.topics:
            return ("This company covered this topic in a prior round — build on "
                    "it with new depth or an example instead of repeating basics.")
        covered = prior_topics(company, role=role, top=5, path=path)
        if covered:
            return ("Prior rounds with this company covered: "
                    + ", ".join(covered) + ". Avoid re-covering them shallowly.")
        return ""
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["RoundLink", "record_topic", "start_round", "prior_link",
           "prior_topics", "link_directive"]
