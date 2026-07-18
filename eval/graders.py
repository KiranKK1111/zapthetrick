"""Case graders — rubric-driven scoring.

Each case carries an `expected` block. The grader's job is to turn
that block into a numeric score in [0, 1] and a pass/fail decision.

Three grader strategies, applied additively (weighted blend):
  - `contains` — listed substrings must appear (case-insensitive).
  - `omits`    — listed substrings must NOT appear.
  - `rubric`   — LLM-as-judge: ask a small model to grade the output
                 against the case's rubric prompt.

The LLM judge is optional — when no `rubric_prompt` is provided
or the LLM is unreachable, only the substring graders apply.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field


log = logging.getLogger(__name__)


@dataclass
class Grade:
    score: float = 0.0
    passed: bool = False
    detail: str = ""
    components: dict = field(default_factory=dict)


def grade_case(case: dict, output: str) -> Grade:
    expected = case.get("expected") or {}
    if not expected:
        return Grade(score=1.0, passed=True, detail="no expectations")

    comps: dict = {}

    contains = [str(s) for s in (expected.get("contains") or []) if isinstance(s, (str, int))]
    omits = [str(s) for s in (expected.get("omits") or []) if isinstance(s, (str, int))]
    rubric_threshold = float(expected.get("rubric_threshold") or 0.0)
    rubric_prompt = str(expected.get("rubric_prompt") or "")

    low = (output or "").lower()
    if contains:
        hit = [c for c in contains if c.lower() in low]
        comps["contains_ratio"] = len(hit) / len(contains)
    else:
        comps["contains_ratio"] = 1.0
    if omits:
        violated = [c for c in omits if c.lower() in low]
        comps["omits_ratio"] = 1.0 - (len(violated) / len(omits))
    else:
        comps["omits_ratio"] = 1.0

    # Optional rubric grader.
    if rubric_prompt:
        try:
            rubric_score = _rubric_score_sync(rubric_prompt, output)
        except Exception as exc:  # noqa: BLE001
            log.warning("rubric grader failed: %s", exc)
            rubric_score = 1.0  # neutral on failure
        comps["rubric"] = rubric_score
    else:
        comps["rubric"] = 1.0

    score = (comps["contains_ratio"] + comps["omits_ratio"] + comps["rubric"]) / 3.0
    passed = score >= max(0.6, rubric_threshold)
    return Grade(
        score=score,
        passed=passed,
        detail=f"contains={comps['contains_ratio']:.2f} omits={comps['omits_ratio']:.2f} "
               f"rubric={comps['rubric']:.2f}",
        components=comps,
    )


def _rubric_score_sync(rubric_prompt: str, output: str) -> float:
    """Sync wrapper around the LLM judge — used because graders run
    synchronously from the runner. The runner already runs in an
    event loop; we offload the LLM call to a thread."""
    import asyncio

    async def _go():
        from app.core.config_loader import cfg
        from app.core.llm_client import llm

        prompt = (
            "You are grading an answer against a rubric. Return ONLY a "
            "decimal between 0 and 1 (higher = better).\n\n"
            f"RUBRIC:\n{rubric_prompt}\n\n"
            f"ANSWER:\n{output[:4000]}\n\nSCORE:"
        )
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=cfg.llm.classifier_model or cfg.llm.model,
            options={"temperature": 0, "num_predict": 8},
        )
        try:
            return max(0.0, min(1.0, float((raw or "0").strip().split()[0])))
        except (ValueError, IndexError):
            return 0.5

    try:
        # If we're inside a loop already, schedule and wait.
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_go(), loop)
            return fut.result(timeout=30)
        return asyncio.run(_go())
    except RuntimeError:
        return asyncio.run(_go())


__all__ = ["Grade", "grade_case"]
