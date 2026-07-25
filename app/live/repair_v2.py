"""Domain transcript repair v2 (vNext §4.11, Stage 6 Component H).

Live STT gets domain jargon wrong in predictable ways ("cube control" → kubectl,
"post grey sequel" → PostgreSQL, "grpz" → gRPC). §4.11 stacks six conservative
layers to fix them without ever mangling a real word; the phonetic + edit-
distance core already lives in `repair.py`, so this module adds the two layers it
was missing and stacks them:

  * an **IT/CS lexicon** (a seed here, extensible to the ~50k baked asset) merged
    with the interview's own resume/JD/org vocabulary, so the repair vocab covers
    both general tech terms and this candidate's specifics;
  * a per-session **correction MEMORY** — once a mis-transcription is repaired,
    the (wrong → right) mapping is remembered and reapplied INSTANTLY and
    CONSISTENTLY for the rest of the session (faster than re-deriving it, and it
    can't flip-flop between turns).

`repair_v2(text, session_id, domain_vocab)` runs: correction-memory pass → the
existing conservative phonetic repair over the merged vocab → LEARN the new
fixes into memory. Substitution-only + fail-open (any error returns the input
unchanged). Flag-gated (`live.repair_v2`, default OFF → today's `repair.repair`).
"""
from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger(__name__)

_LOCK = threading.RLock()

# A small IT/CS lexicon seed. The full ~50k asset (with phonetic keys + spoken
# variants) loads from a data file when present; this keeps the layer useful
# with no asset baked. Lowercase canonical forms.
_SEED_LEXICON: tuple[str, ...] = (
    "kubernetes", "kubectl", "docker", "kafka", "grpc", "graphql", "postgresql",
    "postgres", "redis", "nginx", "terraform", "ansible", "prometheus", "grafana",
    "elasticsearch", "rabbitmq", "cassandra", "mongodb", "dynamodb", "kinesis",
    "lambda", "kotlin", "golang", "typescript", "javascript", "pytorch",
    "tensorflow", "numpy", "pandas", "sqlalchemy", "fastapi", "django", "flask",
    "spring", "hibernate", "webpack", "kubeadm", "istio", "envoy", "grpcurl",
    "oauth", "openid", "jwt", "websocket", "webrtc", "protobuf", "avro",
)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "repair_v2", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# IT/CS lexicon (seed + optional baked asset)
# --------------------------------------------------------------------------- #
_lexicon_cache: list[str] | None = None


def _load_lexicon_asset() -> list[str]:
    """Best-effort load of the baked ~50k IT/CS lexicon (one term per line). '' /
    absent → []. Cached after the first read."""
    try:
        import pathlib
        p = pathlib.Path(__file__).with_name("assets_lexicon_itcs.txt")
        if not p.exists():
            return []
        return [ln.strip().lower() for ln in p.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


def lexicon_terms() -> list[str]:
    """The IT/CS lexicon — seed + baked asset (deduped)."""
    global _lexicon_cache
    if _lexicon_cache is None:
        seen: set[str] = set()
        out: list[str] = []
        for t in list(_SEED_LEXICON) + _load_lexicon_asset():
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        _lexicon_cache = out
    return _lexicon_cache


# --------------------------------------------------------------------------- #
# Per-session correction memory (layer 5)
# --------------------------------------------------------------------------- #
_corrections: dict[str, dict[str, str]] = {}


def remember(session_id: str, wrong: str, right: str) -> None:
    """Record a (wrong → right) correction for a session. Ignores no-ops and
    over-long tokens. Never raises."""
    try:
        raw = (wrong or "").strip()
        w = raw.lower()
        r = (right or "").strip()
        # Reject identity (exact, case-sensitive) — but a case-only fix like
        # "go" → "Go" or "grpc" → "gRPC" IS a real correction and is kept.
        if not w or not r or raw == r or len(w) > 60:
            return
        with _LOCK:
            _corrections.setdefault(session_id or "", {})[w] = r
    except Exception:  # noqa: BLE001
        pass


def corrections(session_id: str) -> dict[str, str]:
    with _LOCK:
        return dict(_corrections.get(session_id or "", {}))


def apply_memory(session_id: str, text: str) -> str:
    """Reapply this session's learned corrections to `text` (whole-word,
    case-insensitive, preserving the original token's leading capitalization).
    Never raises."""
    try:
        mem = corrections(session_id)
        if not mem or not (text or "").strip():
            return text or ""

        def _sub(m: re.Match) -> str:
            tok = m.group(0)
            rep = mem.get(tok.lower())
            if rep is None:
                return tok
            # Preserve the stored casing (gRPC, PostgreSQL) EXCEPT when the token
            # was capitalized (sentence start) and the stored form is all-lower —
            # then just uppercase the first char, don't clobber internal caps.
            if tok[:1].isupper() and rep.islower():
                return rep[:1].upper() + rep[1:]
            return rep

        return re.sub(r"[A-Za-z][A-Za-z'-]*", _sub, text)
    except Exception:  # noqa: BLE001
        return text or ""


def forget_session(session_id: str) -> None:
    with _LOCK:
        _corrections.pop(session_id or "", None)


def reset_for_tests() -> None:
    global _lexicon_cache
    with _LOCK:
        _corrections.clear()
    _lexicon_cache = None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def repair_v2(text: str, *, session_id: str = "", domain_vocab=None) -> str:
    """Stacked domain repair. Off → returns `text` unchanged (the caller keeps
    today's path). On: correction-memory pass → conservative phonetic repair over
    the IT/CS lexicon + domain vocab → LEARN the new fixes into session memory.
    Substitution-only + fail-open."""
    try:
        if not enabled():
            return text or ""
        original = text or ""
        if not original.strip():
            return original
        # Layer 5 first — a known fix is instant + authoritative.
        pre = apply_memory(session_id, original)
        # Layers 1-3 (existing): conservative phonetic / edit-distance repair over
        # the merged vocab (lexicon + this interview's resume/JD/org terms).
        vocab = list(lexicon_terms())
        if domain_vocab:
            vocab += [str(v) for v in domain_vocab]
        from app.live.repair import repair as _repair
        fixed = _repair(pre, vocab=vocab)
        # Learn: any token the repair CHANGED becomes a session correction so a
        # repeat is fixed instantly (and consistently) next turn.
        if session_id:
            _learn_diffs(session_id, pre, fixed)
        return fixed
    except Exception:  # noqa: BLE001
        return text or ""


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _learn_diffs(session_id: str, before: str, after: str) -> None:
    try:
        wb = _WORD.findall(before)
        wa = _WORD.findall(after)
        if len(wb) != len(wa):
            return  # structure changed (phrase pass) — don't guess token pairs
        for b, a in zip(wb, wa):
            if b.lower() != a.lower():
                remember(session_id, b, a)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["enabled", "lexicon_terms", "remember", "corrections",
           "apply_memory", "forget_session", "repair_v2", "reset_for_tests"]
