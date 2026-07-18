"""Runnable eval entrypoint (P2-2).

    python -m app.eval                 # offline regression suite (no keys)
    python -m app.eval --models        # + grade the live routed model on the
                                       #   IT/CS task bank (needs provider keys)
    python -m app.eval --models --category architecture --difficulty hard
    python -m app.eval --models --json report.json

The offline suite always runs (a fast regression guard). With `--models`, the
graded task bank is scored against the app's routed `LLMClient` and a per-
category / per-difficulty scoreboard is printed (and optionally written to
JSON), turning Claude-parity from an estimate into a number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _run_offline() -> bool:
    from app.eval.harness import default_suite, run_suite

    rep = run_suite(default_suite())
    print(f"\n[offline regression] {rep.passed}/{rep.total} passed "
          f"(pass_rate={rep.pass_rate})")
    for r in rep.results:
        if not r.passed:
            print(f"  FAIL {r.category}/{r.name}: {r.detail}")
    return rep.failed == 0


async def _run_models(args) -> dict:
    from app.eval.model_eval import make_llm_runner, run_task_bank
    from app.eval.task_bank import task_bank

    tasks = task_bank(
        categories=args.category or None,
        difficulties=args.difficulty or None,
    )
    if not tasks:
        print("No tasks match the given filters.")
        return {}
    label = args.label or "routed"
    print(f"\n[model eval] grading {len(tasks)} tasks on '{label}' "
          f"(concurrency={args.concurrency})...")
    runner = make_llm_runner()
    board = await run_task_bank(tasks, runner, model=label,
                                concurrency=args.concurrency)
    print(f"\n== {board.model}: {board.passed}/{board.total} passed  "
          f"pass_rate={board.pass_rate}  mean_score={board.mean_score}")
    print("\n  by category:")
    for cat, st in board.by_category().items():
        print(f"    {cat:18s} {st['passed']}/{st['count']:>2}  "
              f"mean={st['mean_score']}")
    print("\n  by difficulty:")
    for diff, st in board.by_difficulty().items():
        print(f"    {diff:10s} {st['passed']}/{st['count']:>2}  "
              f"mean={st['mean_score']}")
    failed = [s for s in board.scores if not s.passed]
    if failed:
        print(f"\n  {len(failed)} below threshold:")
        for s in failed:
            why = s.error or ", ".join(
                g.name for g in s.gate_results if not g.passed)
            print(f"    {s.task_id} (score={s.score}): {why}")
    return board.to_dict()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.eval",
                                description="ZapTheTrick eval harness")
    p.add_argument("--models", action="store_true",
                   help="grade the live routed model on the IT/CS task bank")
    p.add_argument("--category", action="append", default=[],
                   help="filter task bank by category (repeatable)")
    p.add_argument("--difficulty", action="append", default=[],
                   help="filter task bank by difficulty (repeatable)")
    p.add_argument("--label", default="", help="label for the scoreboard")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--json", default="", help="write the model scoreboard JSON")
    args = p.parse_args(argv)

    offline_ok = _run_offline()

    if args.models:
        board = asyncio.run(_run_models(args))
        if args.json and board:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(board, fh, indent=2)
            print(f"\nwrote {args.json}")

    return 0 if offline_ok else 1


if __name__ == "__main__":
    sys.exit(main())
