"""Automatic test generation (agent-orchestration R5).

`generate_and_run(workspace_id, change, gen_fn, test_cmd, runner)` lets the
workflow (optionally) generate tests for produced code, run them in the
execution sandbox, and report pass/fail — feeding the
`evaluation-and-reliability` confidence/quality signals via `confidence_delta`
(R5). Disabled → skipped, today's behavior (R5.4, Property 5). Injectable +
async; never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from app.orchestration.sandbox import run_code, SandboxResult


@dataclass
class TestGenResult:
    generated: bool
    ran: bool
    passed: bool
    status: str               # "skipped" | "no_tests" | "passed" | "failed" | "error"
    detail: str = ""

    @property
    def confidence_delta(self) -> float:
        """Signal fed to the evaluation aggregate confidence: + on pass, - on
        fail, 0 when not run (R5.3)."""
        if not self.ran:
            return 0.0
        return 0.15 if self.passed else -0.2


def _enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.orchestration, "generate_tests", False))
    except Exception:  # noqa: BLE001
        return False


async def generate_and_run(
    workspace_id: str,
    change: str,
    *,
    gen_fn: Callable[[str], Awaitable[str]] | None = None,
    test_cmd: str = "",
    runner: Callable[..., Awaitable] | None = None,
    force: bool = False,
) -> TestGenResult:
    """Generate tests for `change` via `gen_fn` (injected; an LLM pass in prod),
    then run `test_cmd` in the sandbox. Disabled / no generator → skipped.
    Never raises."""
    try:
        if not force and not _enabled():
            return TestGenResult(False, False, False, "skipped")
        if gen_fn is None:
            return TestGenResult(False, False, False, "skipped",
                                 "no test generator")
        tests = await gen_fn(change)
        if not (tests or "").strip():
            return TestGenResult(False, False, False, "no_tests")
        if not test_cmd.strip():
            # Generated but nothing to run them with → reported, not verified.
            return TestGenResult(True, False, False, "no_tests",
                                 "tests generated; no run command")
        res: SandboxResult = await run_code(workspace_id, test_cmd,
                                            runner=runner, force=True)
        if not res.ran:
            return TestGenResult(True, False, False, res.status,
                                 res.repair_feedback)
        if res.verified:
            return TestGenResult(True, True, True, "passed", res.output)
        return TestGenResult(True, True, False, "failed", res.repair_feedback)
    except Exception as exc:  # noqa: BLE001
        return TestGenResult(False, False, False, "error", str(exc))


__all__ = ["generate_and_run", "TestGenResult"]
