"""Red-team / adversarial review (Phase 8, report #112/#113).

After the agent produces a result, a SECOND pass plays devil's advocate: it
hunts for correctness bugs, security holes, missing error handling, and edge
cases the build/tests wouldn't catch. Returns structured `Risk`s the UI shows
as a "Review" card and that feed the confidence band.

One fast JSON LLM call; provider-agnostic; never raises (returns [] on any
failure) so it can't break a run. Best used as an advisory layer, not a gate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

log = logging.getLogger(__name__)

_SEVERITIES = ("high", "medium", "low")
_MAX_RISKS = 8

_PROMPT = (
    "You are a STRICT senior reviewer doing an adversarial (red-team) review of "
    "another engineer's work. Find concrete problems a passing build/test run "
    "would MISS: correctness bugs, security vulnerabilities (injection, authz, "
    "secrets, unsafe input), missing error handling, race conditions, resource "
    "leaks, and unhandled edge cases. Be specific and actionable; do NOT invent "
    "issues — if the work looks sound, return very few or none.\n\n"
    "Reply with ONLY compact JSON, no prose:\n"
    "{\"risks\": [{\"severity\": \"high|medium|low\", \"area\": \"<short tag, "
    "e.g. security/correctness/edge-case/error-handling>\", \"issue\": \"<one "
    "sentence>\", \"fix\": \"<one-sentence suggested fix>\"}]}\n\n"
    "TASK:\n{task}\n\nWORK PRODUCED (summary / diff / answer):\n{work}\n"
)


@dataclass
class Risk:
    severity: str
    area: str
    issue: str
    fix: str = ""


def _parse(raw: str) -> list[Risk]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    items = obj.get("risks") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out: list[Risk] = []
    for it in items[:_MAX_RISKS]:
        if not isinstance(it, dict):
            continue
        issue = str(it.get("issue") or "").strip()
        if not issue:
            continue
        sev = str(it.get("severity") or "medium").strip().lower()
        if sev not in _SEVERITIES:
            sev = "medium"
        out.append(Risk(
            severity=sev,
            area=str(it.get("area") or "general").strip()[:32],
            issue=issue[:300],
            fix=str(it.get("fix") or "").strip()[:300],
        ))
    return out


async def red_team_review(task: str, work: str) -> list[Risk]:
    """Adversarially review `work` for `task`. [] on empty input or any failure."""
    if not (work or "").strip():
        return []
    from app.core.prompt import fill
    prompt = fill(_PROMPT, task=(task or "")[:2000], work=work[:8000])
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=(cfg.llm.code_model or cfg.llm.model),
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.redteam},
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001
        log.info("red-team review failed (skipping): %s", exc)
        return []
    return _parse(raw)


def risks_to_dicts(risks: list[Risk]) -> list[dict]:
    return [asdict(r) for r in risks]


def count_high(risks: list[Risk]) -> int:
    return sum(1 for r in risks if r.severity == "high")


__all__ = ["Risk", "red_team_review", "risks_to_dicts", "count_high"]
