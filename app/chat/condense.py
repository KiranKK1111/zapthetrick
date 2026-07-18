"""Within-message cap: condense one oversized message to its important lines.

The composer collapses big pastes into chips, but the user can still SEND a
message whose body is enormous (a whole file, a giant log, a million-line
paste). [window_messages] trims across *turns*, not within one — it always
keeps the latest message verbatim, so a single 40 MB paste would still be
shipped whole and blow the model's context window.

`condense_oversized` is the per-message guard. Instead of rejecting an oversized
body it KEEPS the most important lines: the head and tail verbatim (an
instruction/question usually sits at the very start or end of the pasted body,
and code/logs need their boundaries) plus the highest-salience lines from the
middle — skeleton lines like ``def`` / ``class`` / ``error`` / headings —
dropping noise and collapsing the gaps into ``[… N lines omitted …]`` markers.

No LLM call: a cheap, deterministic salience pass, so it stays fast even on
millions of lines. The full original is still stored on the Message row; only
the copy shipped to the model is reduced.
"""
from __future__ import annotations

import re

# A single message shipped to the model is capped at roughly this many chars
# (~15k tokens) — generous for a focused paste, small enough to leave room for
# the rest of the conversation inside the history window budget.
MAX_MESSAGE_CHARS = 60_000

# Head/tail kept verbatim, bounded by BOTH a char budget and a line count — the
# line cap stops a run of blank/tiny lines from packing thousands into the head.
_HEAD_CHARS = 8_000
_TAIL_CHARS = 8_000
_HEAD_LINES = 60
_TAIL_LINES = 60
# Reserve for the one-line banner prepended to the result, so the returned
# string stays within max_chars.
_BANNER_RESERVE = 220
# Any single line longer than this is itself truncated, so one 2 MB minified
# line can't swallow the whole budget.
_MAX_LINE_CHARS = 2_000
# Each kept middle line may be preceded by an "[… N omitted …]" marker; reserve
# this many chars per kept line so the markers don't blow past the budget.
_MARKER_RESERVE = 32


def _truncate_line(line: str) -> str:
    if len(line) <= _MAX_LINE_CHARS:
        return line
    return line[:_MAX_LINE_CHARS] + f" …[+{len(line) - _MAX_LINE_CHARS} chars]"


_DIGITS = re.compile(r"\d+")


def _norm(line: str) -> str:
    """Whitespace-normalised, bounded form of a line (so a 2 MB line doesn't
    cost 2 MB to process). Language-agnostic — no vocabulary."""
    return " ".join(line[:_MAX_LINE_CHARS].split())


def _template(norm: str) -> int:
    """Signature of a line's STRUCTURE — same as [_norm] but with digit runs
    masked, so "tick 1" / "tick 9999" (and every log line that differs only by
    an id/number/timestamp) collapse to one template. Lets distinct structures
    (an error line, a definition) surface even when each instance is unique."""
    return hash(_DIGITS.sub("#", norm))


def condense_oversized(
    text: str | None, *, max_chars: int = MAX_MESSAGE_CHARS
) -> tuple[str | None, bool]:
    """Return ``(content, was_condensed)``.

    If ``text`` is within ``max_chars`` it's returned unchanged. Otherwise it's
    reduced to its most important lines under (roughly) ``max_chars`` — head and
    tail verbatim, the best middle lines kept, gaps marked.
    """
    if not text or len(text) <= max_chars:
        return text, False

    lines = text.splitlines()
    n = len(lines)

    # Budgets are relative to the cap (so a small max_chars can't be blown by
    # the absolute head/tail constants) and leave room for the banner.
    content_budget = max(200, max_chars - _BANNER_RESERVE)
    head_budget = min(_HEAD_CHARS, content_budget // 3)
    tail_budget = min(_TAIL_CHARS, content_budget // 3)

    # 1) Head kept verbatim: first lines, capped by chars AND line count.
    head: list[str] = []
    used = 0
    hi = 0
    while hi < n and len(head) < _HEAD_LINES and used < head_budget:
        ln = _truncate_line(lines[hi])
        head.append(ln)
        used += len(ln) + 1
        hi += 1

    # 2) Tail kept verbatim (never overlapping the head), same dual cap.
    tail: list[str] = []
    used_t = 0
    ti = n - 1
    while ti >= hi and len(tail) < _TAIL_LINES and used_t < tail_budget:
        ln = _truncate_line(lines[ti])
        tail.append(ln)
        used_t += len(ln) + 1
        ti -= 1
    tail.reverse()
    tail_start = ti + 1  # first index belonging to the tail

    # 3) Middle: language-agnostic, keyword-free selection. Drop blank lines and
    #    exact duplicates, then (a) keep an EVENLY SPACED sample so coverage is
    #    spread across the whole body, and (b) ensure each distinct line
    #    STRUCTURE (digit-masked template) is represented, so a rare error /
    #    definition line surfaces even though it isn't "important" by any
    #    vocabulary. No salience scoring, no keywords.
    mids = range(hi, tail_start)
    middle_budget = content_budget - used - used_t

    seen_exact: set[int] = set()
    seen_tmpl: set[int] = set()
    candidates: list[int] = []
    tmpl_firsts: list[int] = []
    for i in mids:
        if not lines[i].strip():
            continue  # blank — represented by the omission markers
        norm = _norm(lines[i])
        sx = hash(norm)
        if sx in seen_exact:
            continue  # exact (whitespace-normalised) duplicate
        seen_exact.add(sx)
        candidates.append(i)
        st = _template(norm)
        if st not in seen_tmpl:
            seen_tmpl.add(st)
            tmpl_firsts.append(i)

    def _cost(idx: int) -> int:
        return min(len(lines[idx]), _MAX_LINE_CHARS) + 1 + _MARKER_RESERVE

    total_cost = sum(_cost(i) for i in candidates)
    if total_cost <= middle_budget or not candidates:
        sampled = list(candidates)
        target = len(candidates)
    else:
        avg = max(1, total_cost // len(candidates))
        target = max(1, middle_budget // avg)
        stride = len(candidates) / target
        picked = {candidates[min(len(candidates) - 1, int(t * stride))]
                  for t in range(target)}
        sampled = [i for i in candidates if i in picked]

    # Add the distinct structures — but only when they're FEW (repetitive
    # content like logs); when nearly every line is its own template (diverse
    # prose/code) the even sample already covers it, so skip to avoid biasing
    # toward the front.
    template_reps = tmpl_firsts if len(tmpl_firsts) <= target else []

    keep: set[int] = set()
    spent = 0
    for i in sorted(set(sampled) | set(template_reps)):
        cost = _cost(i)
        if spent + cost > middle_budget:
            break
        keep.add(i)
        spent += cost

    # 4) Reassemble in original order, collapsing dropped runs into one marker.
    out: list[str] = list(head)
    gap = 0
    for i in mids:
        if i in keep:
            if gap:
                out.append(f"[… {gap:,} line{'s' if gap != 1 else ''} omitted …]")
                gap = 0
            out.append(_truncate_line(lines[i]))
        else:
            gap += 1
    if gap:
        out.append(f"[… {gap:,} line{'s' if gap != 1 else ''} omitted …]")
    out.extend(tail)

    banner = (
        f"[Note: a large paste of {n:,} lines was condensed to fit the model's "
        f"context - the head and tail are kept verbatim with an evenly-spaced "
        f"sample of the rest (duplicates dropped); omitted runs are marked.]"
    )
    return banner + "\n" + "\n".join(out), True


__all__ = ["condense_oversized", "MAX_MESSAGE_CHARS"]
