"""Model-comparison eval runner + scoreboard + optional LLM judge (P2-2).

This is the layer that turns the graded task bank into a *measurement*:

  - `run_task_bank(tasks, runner)` — runs every task through an async `runner`
    (which produces the model's text answer) and grades it with the task's
    objective gates, aggregating into a `Scoreboard` (overall + per-category +
    per-difficulty pass rate / mean score).
  - `compare_models(tasks, runners)` — same, for several named runners, so two
    models (e.g. a free model vs a strong tier) are scored on the identical
    rubric and ranked.
  - `judge_ranking(task, candidates, judge)` — the SECOND axis: an LLM judge
    ranks several candidates' answers for a task (pairwise/listwise), used to
    capture quality the objective gates can't (clarity, correctness nuance).
    The judge is an injected async callable, so it's model-agnostic and mocked
    in tests.
  - `make_llm_runner()` — wires a real runner onto the app's `LLMClient`, so
    `python -m app.eval --models` measures the live routed model(s).

Everything is provider-agnostic: pass any `async (GradedTask) -> str` runner.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.chat.difficulty import EXPERT
from app.eval.scoring import GradedTask, TaskScore, grade_output

# An async runner takes a task and returns the model's answer text.
Runner = Callable[[GradedTask], Awaitable[str]]
# A judge takes a prompt and returns the model's text (for ranking).
Judge = Callable[[str], Awaitable[str]]


@dataclass
class Scoreboard:
    model: str
    scores: list[TaskScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 3) if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return round(sum(s.score for s in self.scores) / self.total, 3) \
            if self.total else 0.0

    def _group(self, key: str) -> dict[str, dict]:
        out: dict[str, list[TaskScore]] = {}
        for s in self.scores:
            out.setdefault(getattr(s, key), []).append(s)
        return {
            k: {
                "count": len(v),
                "passed": sum(1 for s in v if s.passed),
                "pass_rate": round(sum(1 for s in v if s.passed) / len(v), 3),
                "mean_score": round(sum(s.score for s in v) / len(v), 3),
            }
            for k, v in sorted(out.items())
        }

    def by_category(self) -> dict[str, dict]:
        return self._group("category")

    def by_difficulty(self) -> dict[str, dict]:
        return self._group("difficulty")

    def to_dict(self, *, include_tasks: bool = True) -> dict:
        d = {
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "by_category": self.by_category(),
            "by_difficulty": self.by_difficulty(),
        }
        if include_tasks:
            d["tasks"] = [s.to_dict() for s in self.scores]
        return d


async def run_task_bank(
    tasks: list[GradedTask],
    runner: Runner,
    *,
    model: str = "model",
    pass_threshold: float = 0.7,
    concurrency: int = 4,
) -> Scoreboard:
    """Run + grade every task through `runner` (bounded concurrency)."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(task: GradedTask) -> TaskScore:
        async with sem:
            try:
                output = await runner(task)
                return grade_output(task, output, pass_threshold=pass_threshold)
            except Exception as exc:  # noqa: BLE001 — a model error fails the task
                return grade_output(task, "", pass_threshold=pass_threshold,
                                    error=f"{type(exc).__name__}: {exc}")

    scores = await asyncio.gather(*(_one(t) for t in tasks))
    return Scoreboard(model=model, scores=list(scores))


async def compare_models(
    tasks: list[GradedTask],
    runners: dict[str, Runner],
    *,
    pass_threshold: float = 0.7,
    concurrency: int = 4,
) -> dict[str, Scoreboard]:
    """Score each named runner on the same tasks. Keys = model labels."""
    out: dict[str, Scoreboard] = {}
    for name, runner in runners.items():
        out[name] = await run_task_bank(
            tasks, runner, model=name, pass_threshold=pass_threshold,
            concurrency=concurrency,
        )
    return out


def rank_models(boards: dict[str, Scoreboard]) -> list[dict]:
    """A leaderboard sorted by mean_score then pass_rate (desc)."""
    rows = [
        {"model": b.model, "mean_score": b.mean_score,
         "pass_rate": b.pass_rate, "passed": b.passed, "total": b.total}
        for b in boards.values()
    ]
    rows.sort(key=lambda r: (r["mean_score"], r["pass_rate"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ── optional LLM judge (quality axis the objective gates can't capture) ─────
_JUDGE_PROMPT = """You are grading answers to a software engineering task.

TASK:
{task}

{reference}CANDIDATES:
{candidates}

Rank the candidates from best to worst on correctness, completeness, and
clarity. Reply with ONLY a JSON object:
{{"ranking": ["<label>", ...], "winner": "<label>", "reason": "<one line>"}}
"""


def _parse_judge(text: str, labels: list[str]) -> dict:
    """Parse the judge's JSON verdict; degrade gracefully on malformed output."""
    m = re.search(r"\{.*\}", text or "", re.S)
    raw = m.group(0) if m else (text or "")
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        # fall back: first label mentioned wins
        low = (text or "").lower()
        winner = next((l for l in labels if l.lower() in low), None)
        return {"ranking": [winner] if winner else [], "winner": winner,
                "reason": "unparseable judge output", "raw": text}
    ranking = [l for l in data.get("ranking", []) if l in labels]
    winner = data.get("winner") if data.get("winner") in labels else \
        (ranking[0] if ranking else None)
    return {"ranking": ranking, "winner": winner,
            "reason": str(data.get("reason", ""))[:200]}


async def judge_ranking(
    task: GradedTask,
    candidates: dict[str, str],
    judge: Judge,
) -> dict:
    """Have an LLM judge rank candidate answers for one task.

    `candidates` maps a label (e.g. model name) to its answer. `judge` is an
    async callable that takes a prompt and returns text. Returns a parsed
    verdict {ranking, winner, reason}.
    """
    ref = f"REFERENCE (gold) ANSWER:\n{task.reference}\n\n" if task.reference else ""
    body = "\n\n".join(
        f"[{label}]\n{(text or '').strip()[:2000]}"
        for label, text in candidates.items()
    )
    prompt = _JUDGE_PROMPT.format(task=task.prompt, reference=ref,
                                  candidates=body)
    verdict_text = await judge(prompt)
    return _parse_judge(verdict_text, list(candidates.keys()))


async def judge_compare(
    tasks: list[GradedTask],
    candidate_boards: dict[str, dict[str, str]],
    judge: Judge,
) -> dict:
    """Run the LLM judge across many tasks; tally per-model wins.

    `candidate_boards` maps task_id -> {label -> answer}. Returns win counts
    plus per-task verdicts.
    """
    wins: dict[str, int] = {}
    verdicts: list[dict] = []
    for task in tasks:
        cands = candidate_boards.get(task.id)
        if not cands:
            continue
        v = await judge_ranking(task, cands, judge)
        if v.get("winner"):
            wins[v["winner"]] = wins.get(v["winner"], 0) + 1
        verdicts.append({"task_id": task.id, **v})
    return {"wins": wins, "verdicts": verdicts}


# ── real runner wired onto the app's LLMClient ──────────────────────────────
def make_llm_runner(*, model: str | None = None,
                    max_tokens: int = 1200) -> Runner:
    """A runner that answers each task via the app's routed `LLMClient`.

    Uses the task's `difficulty` so hard/expert tasks exercise the same
    capability-aware routing (and opt-in strong tier from P2-1) the real chat
    path uses. Provider-agnostic.
    """
    from app.core.llm_client import llm

    async def _run(task: GradedTask) -> str:
        messages = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": task.prompt})
        options = {"difficulty": task.difficulty, "num_predict": max_tokens,
                   "max_tokens": max_tokens}
        return await llm.complete(messages, model, options)

    return _run


def make_llm_judge(*, model: str | None = None) -> Judge:
    """An LLM judge wired onto the app's `LLMClient` (expert difficulty)."""
    from app.core.llm_client import llm

    async def _judge(prompt: str) -> str:
        return await llm.complete(
            [{"role": "user", "content": prompt}], model,
            {"difficulty": EXPERT, "temperature": 0.0},
        )

    return _judge


__all__ = [
    "Scoreboard", "Runner", "Judge",
    "run_task_bank", "compare_models", "rank_models",
    "judge_ranking", "judge_compare",
    "make_llm_runner", "make_llm_judge",
]
