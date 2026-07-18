"""Evaluation harness — a free-model / offline regression benchmark plus a
graded Claude-comparison harness (Phase 9 + P2-2).

Two layers:
  - `harness` — the original offline regression suite over the app's
    deterministic decision components (always runnable, no keys).
  - `scoring` / `task_bank` / `model_eval` — a graded IT/CS task bank scored by
    OBJECTIVE gates, a per-model `Scoreboard`, and an optional LLM judge, so
    parity is *measured* per model rather than estimated.

Run the measurement from the CLI:  `python -m app.eval`            (offline)
                                    `python -m app.eval --models`   (live model)
"""
from .harness import EvalCase, EvalReport, EvalResult, default_suite, run_suite
from .model_eval import (
    Scoreboard,
    compare_models,
    judge_compare,
    judge_ranking,
    make_llm_judge,
    make_llm_runner,
    rank_models,
    run_task_bank,
)
from .prompt_eval import (
    Comparison,
    PromptCase,
    PromptEvalResult,
    PromptRegistry,
    PromptVariant,
    Verdict,
    compare_variants,
    evaluate_prompt,
)
from .scoring import GradedTask, Gate, TaskScore, grade_output
from .task_bank import categories, default_task_bank, task_bank

__all__ = [
    # offline regression
    "EvalCase", "EvalResult", "EvalReport", "default_suite", "run_suite",
    # graded model comparison
    "Gate", "GradedTask", "TaskScore", "grade_output",
    "default_task_bank", "task_bank", "categories",
    "Scoreboard", "run_task_bank", "compare_models", "rank_models",
    "judge_ranking", "judge_compare", "make_llm_runner", "make_llm_judge",
    # prompt evaluation (Phase 1 #7)
    "PromptVariant", "PromptCase", "PromptRegistry", "PromptEvalResult",
    "evaluate_prompt", "compare_variants", "Comparison", "Verdict",
]
