"""Live coding two-lane gate (vNext §4.4, Stage 6 Component G).

A live coding answer has two very different latency profiles. The **prose lane**
— the approach, the complexity, the edge cases — is dictatable immediately (TTFT
≤900 ms). The **code lane** must NOT be shown until it has actually compiled and
run in the sandbox (a wrong-but-confident block is worse than a brief wait), so
it's held server-side, verified (tree-sitter → warm-pool compile/run → repair
≤2), then revealed with an honest badge — or, if the sandbox can't run it, shown
with a truthful "not executed" note rather than a false "verified".

This module owns the reusable, deterministic pieces of that gate:
  * `split_lanes(answer)` — separate the streamable PROSE from the held CODE;
  * `complexity_note(code)` — a static Big-O cross-check (advisory only);
  * `reveal_badge(status, …)` — the honest label ("✅ compiled & ran" /
    "⚠ fixed a runtime error" / "ℹ not executed");
  * `compose(prose, code, language, badge)` — reassemble once the code is verified.

The actual sandbox RUN + regenerate already lives in the WS verifier
(`routes_ws.py` + `app/live/code_run.py`); this adds the lane split + complexity
note. Pure + fail-open. Flag-gated (`live.two_lane`, default OFF → today's
single-lane answer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"```([A-Za-z0-9_+#-]*)\s*\n(.*?)```", re.S)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "two_lane", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class LaneSplit:
    prose: str            # the streamable prose (code fences removed)
    code: str             # the primary code block body (held for verification)
    language: str         # the code block's language tag (normalized)
    has_code: bool


def split_lanes(answer: str) -> LaneSplit:
    """Split a coding answer into the PROSE lane (streamable now) and the CODE
    lane (held until verified). Prose = the answer with fenced code removed and
    collapsed whitespace; code = the LONGEST fenced block. Never raises."""
    try:
        text = answer or ""
        blocks = _FENCE.findall(text)
        if not blocks:
            return LaneSplit(prose=text.strip(), code="", language="",
                             has_code=False)
        tag, body = max(blocks, key=lambda tb: len(tb[1]))
        prose = _FENCE.sub(" ", text)
        prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
        try:
            from app.live.code_run import normalize_lang
            lang = normalize_lang(tag)
        except Exception:  # noqa: BLE001
            lang = (tag or "").strip().lower()
        return LaneSplit(prose=prose, code=body.strip(), language=lang,
                         has_code=bool(body.strip()))
    except Exception:  # noqa: BLE001
        return LaneSplit(prose=(answer or "").strip(), code="", language="",
                         has_code=False)


# --------------------------------------------------------------------------- #
# Static complexity cross-check (advisory)
# --------------------------------------------------------------------------- #
_LOOP_RE = re.compile(r"\b(for|while)\b|\.forEach\(|\bmap\(|\bfilter\(")
_RECUR_HINT = re.compile(r"\breturn\b[^\n]*\b(\w+)\s*\(")


def _max_loop_depth(code: str) -> int:
    """Approximate maximum NESTED loop depth by brace/indent structure. A cheap
    heuristic — advisory, never authoritative."""
    depth = 0
    best = 0
    for line in (code or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if _LOOP_RE.search(s):
            depth += 1
            best = max(best, depth)
        # A closing brace or a dedent to column 0 resets a level (very rough).
        if s in ("}", "};") or (line and not line[0].isspace() and depth > 0
                                 and not _LOOP_RE.search(s)):
            depth = max(0, depth - 1)
    return best


def complexity_note(code: str, language: str = "") -> str:
    """A static, ADVISORY Big-O cross-check for the dictated approach. Never
    authoritative (it's a syntactic heuristic) — the note says so. '' when no
    code / on error."""
    try:
        if not (code or "").strip():
            return ""
        depth = _max_loop_depth(code)
        recursive = bool(_RECUR_HINT.search(code))
        if depth >= 3:
            est = "O(n³) or worse"
        elif depth == 2:
            est = "O(n²)"
        elif depth == 1:
            est = "O(n log n) or O(n)"
        elif recursive:
            est = "recursive — likely O(n) to O(2ⁿ) depending on branching"
        else:
            est = "O(1) / no loops"
        return f"static check (advisory): time complexity looks like ~{est}."
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Honest reveal badge
# --------------------------------------------------------------------------- #
def reveal_badge(status: str, language: str = "", *, examples: int | None = None,
                 repaired: bool = False) -> str:
    """The honest one-line badge for the revealed code. `status` is the sandbox
    verdict; anything the sandbox couldn't run is 'not executed', NEVER
    'verified' (§4.4 honest degradation). Never raises."""
    try:
        lang = (language or "code").strip()
        st = (status or "").strip().lower()
        if st in ("verified", "passed", "ok", "ran"):
            ex = (f", {examples} example{'s' if examples != 1 else ''}"
                  if examples else "")
            base = f"✅ Compiled & ran ({lang}{ex})"
            return base + (" · fixed a runtime error" if repaired else "")
        if st in ("not_run", "unavailable", "disabled", "skipped"):
            return f"ℹ Not executed ({lang}) — sandbox unavailable for this answer"
        if st in ("failed", "error", "gibberish", "leaked"):
            return f"⚠ Not executed ({lang}) — the code did not run cleanly"
        return f"ℹ Not executed ({lang})"
    except Exception:  # noqa: BLE001
        return ""


def compose(prose: str, code: str, language: str, badge: str = "") -> str:
    """Reassemble the verified answer: prose, then the fenced code, then the
    honest badge. Used once the code lane clears the sandbox."""
    try:
        parts = [(prose or "").strip()]
        if (code or "").strip():
            parts.append(f"```{language or ''}\n{code.strip()}\n```")
        if badge:
            parts.append(f"_{badge}_")
        return "\n\n".join(p for p in parts if p).strip()
    except Exception:  # noqa: BLE001
        return (prose or "").strip()


__all__ = ["LaneSplit", "enabled", "split_lanes", "complexity_note",
           "reveal_badge", "compose"]
