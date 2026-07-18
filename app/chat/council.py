"""Multi-provider council / cross-model verification (Phase 12, report B1/B2).

The differentiator a single-model assistant can't offer: after the agent
produces a result, have a DIFFERENT free model — or a COUNCIL of several — judge
independently whether the work satisfies the task. Routing each judge with
`avoid_model_db_id` forces a genuinely different model than the one that drafted
(and than the previous judge), so the verdict is a real cross-model check.

  • `cross_model_verify(task, work, n=1)` — one independent judge (B1).
  • `cross_model_verify(task, work, n=3)` — a 3-model council majority vote (B2).
  • `best_of_n(task, candidates)` — a judge ranks candidate answers and picks
    the best (B2 selection, for callers that generate several drafts).

Provider-agnostic, JSON-parsed, never raises (a failed/again-empty judge just
doesn't vote), so it can't break a run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.chat.difficulty import STANDARD
from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm

log = logging.getLogger(__name__)

_MAX_WORK = 8000
_MAX_ISSUES = 8

_JUDGE_PROMPT = (
    "You are an INDEPENDENT senior reviewer on a DIFFERENT model from the one "
    "that produced this work. Judge ONLY whether the WORK correctly and "
    "completely satisfies the TASK. Be objective: agree only if it's genuinely "
    "correct and complete; otherwise list the specific problems.\n\n"
    "Reply with ONLY compact JSON, no prose:\n"
    "{\"agree\": true|false, \"issues\": [\"<short specific problem>\", ...]}\n\n"
    "TASK:\n{task}\n\nWORK:\n{work}\n"
)


@dataclass
class CouncilVerdict:
    agree: bool = True               # majority verdict
    agreement: float = 1.0           # fraction of judges that agreed
    votes: int = 0                   # judges that actually voted
    verifiers: list[str] = field(default_factory=list)   # model ids/names
    issues: list[str] = field(default_factory=list)      # merged, deduped

    def to_dict(self) -> dict:
        return {
            "agree": self.agree, "agreement": round(self.agreement, 2),
            "votes": self.votes, "verifiers": self.verifiers,
            "issues": self.issues,
        }


def _parse_vote(raw: str) -> tuple[bool | None, list[str]]:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j + 1]
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return None, []
    if not isinstance(obj, dict) or "agree" not in obj:
        return None, []
    issues = obj.get("issues")
    issues = [str(x).strip() for x in issues if str(x).strip()] \
        if isinstance(issues, list) else []
    return bool(obj.get("agree")), issues[:_MAX_ISSUES]


async def cross_model_verify(task: str, work: str, *, n: int = 1,
                             avoid_model_db_id: int | None = None) -> CouncilVerdict:
    """Run `n` independent judges (each forced onto a different model) and
    aggregate a majority verdict. n=1 → cross-model verify; n>1 → council vote.
    Returns a neutral (agree) verdict if no judge could vote."""
    if not (work or "").strip():
        return CouncilVerdict()
    from app.core.prompt import fill

    n = max(1, int(n))
    prompt = fill(_JUDGE_PROMPT, task=(task or "")[:2000], work=work[:_MAX_WORK])
    agree_votes = 0
    total = 0
    verifiers: list[str] = []
    issues: list[str] = []
    avoid = avoid_model_db_id

    for _ in range(n):
        try:
            text, model_db_id = await llm.complete_routed(
                [{"role": "user", "content": prompt}],
                options={"temperature": cfg.temperature.classifier,
                         "num_predict": cfg.output_tokens.council_gen,
                         "difficulty": STANDARD,
                         **({"avoid_model_db_id": avoid} if avoid else {})},
            )
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.info("council judge failed (skipping vote): %s", exc)
            continue
        verdict, vote_issues = _parse_vote(text)
        if verdict is None:
            continue
        total += 1
        if verdict:
            agree_votes += 1
        else:
            for it in vote_issues:
                if it not in issues:
                    issues.append(it)
        if model_db_id is not None:
            verifiers.append(str(model_db_id))
            avoid = model_db_id  # next judge avoids this one → different model

    if total == 0:
        return CouncilVerdict()  # nobody voted → don't penalize
    agreement = agree_votes / total
    return CouncilVerdict(
        agree=(agree_votes * 2 >= total),       # majority (ties → agree)
        agreement=agreement, votes=total,
        verifiers=verifiers, issues=issues[:_MAX_ISSUES],
    )


_PICK_PROMPT = (
    "You are judging {n} candidate answers to the same TASK. Pick the single "
    "BEST one (most correct, complete, and clear). Reply with ONLY compact "
    "JSON: {\"best\": <1-based index>, \"why\": \"<one short reason>\"}.\n\n"
    "TASK:\n{task}\n\n{candidates}\n"
)


async def best_of_n(task: str, candidates: list[str]) -> tuple[int, str]:
    """A judge ranks `candidates` and returns (best_index, rationale). Falls back
    to the first candidate on any failure. Index is 0-based into `candidates`."""
    cands = [c for c in candidates if (c or "").strip()]
    if not cands:
        return 0, "no candidates"
    if len(cands) == 1:
        return 0, "only one candidate"
    from app.core.prompt import fill

    blocks = "\n\n".join(
        f"--- Candidate {i + 1} ---\n{c[:3000]}" for i, c in enumerate(cands))
    prompt = fill(_PICK_PROMPT, n=len(cands), task=(task or "")[:1500],
                  candidates=blocks)
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            options={"temperature": cfg.temperature.classifier,
                     "num_predict": cfg.output_tokens.council_pick,
                     "difficulty": STANDARD})
    except (LLMError, Exception) as exc:  # noqa: BLE001
        log.info("best_of_n judge failed (using first): %s", exc)
        return 0, "judge unavailable"
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j + 1]
    try:
        obj = json.loads(s)
        idx = int(obj.get("best", 1)) - 1
        why = str(obj.get("why") or "").strip()
    except Exception:  # noqa: BLE001
        return 0, "unparseable verdict"
    if idx < 0 or idx >= len(cands):
        idx = 0
    return idx, why


def council_enabled() -> tuple[bool, int]:
    """(enabled, size) from config, safe defaults."""
    try:
        on = bool(getattr(cfg.advanced_rag, "cross_model_verify", True))
        size = max(1, int(getattr(cfg.advanced_rag, "council_size", 1)))
        return on, size
    except Exception:  # noqa: BLE001
        return True, 1


__all__ = ["CouncilVerdict", "cross_model_verify", "best_of_n",
           "council_enabled"]
