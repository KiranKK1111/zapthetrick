"""Verification loop (#32) — detect the build system and actually run the
project's build / tests / lint, then report structured pass/fail with the
failing output so the agent's repair loop can fix it.

    report = await verify_workspace(root, steps=("build", "test"))
    if not report.ok:
        feedback = report.feedback()   # actionable failures for the model

Built on Phase 0: `buildsys.detect_build_system` (#22) + `runner.run_in_workspace`
(#7). Languages whose command is missing (or whose toolchain isn't installed)
are reported as `skipped`, never as failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .buildsys import detect_build_system
from .runner import run_in_workspace

# A toolchain that isn't installed shows up as exit 127 (POSIX) / 9009 (Windows
# "not recognized"). We treat that as SKIPPED, not a real failure.
_NOT_FOUND_CODES = {127, 9009}
_NOT_FOUND_HINTS = ("not found", "not recognized", "no such file",
                    "command not found", "is not recognized")
_TAIL_LINES = 40


@dataclass
class VerifyStep:
    name: str            # build | test | lint
    command: str
    exit_code: int = 0
    ok: bool = False
    skipped: bool = False
    timed_out: bool = False
    output_tail: str = ""


@dataclass
class VerifyReport:
    system: str | None = None
    language: str | None = None
    steps: list[VerifyStep] = field(default_factory=list)

    @property
    def attempted(self) -> bool:
        """True if at least one step actually ran (not skipped)."""
        return any(not s.skipped for s in self.steps)

    @property
    def ok(self) -> bool:
        """All attempted steps passed (a no-build-system workspace is 'ok')."""
        ran = [s for s in self.steps if not s.skipped]
        return all(s.ok for s in ran)

    @property
    def summary(self) -> str:
        if self.system is None:
            return "No recognized build system — nothing to verify."
        lines = [f"Build system: {self.system} ({self.language})"]
        for s in self.steps:
            if s.skipped:
                mark = "skipped"
            elif s.timed_out:
                mark = "TIMED OUT"
            elif s.ok:
                mark = "PASS"
            else:
                mark = f"FAIL (exit {s.exit_code})"
            lines.append(f"  - {s.name}: {mark}  [{s.command}]")
        return "\n".join(lines)

    def feedback(self) -> str:
        """Failing-step detail the repair loop feeds back to the model."""
        bad = [s for s in self.steps
               if not s.skipped and not s.ok]
        if not bad:
            return ""
        out = ["Verification failed — fix these and we'll re-run:"]
        for s in bad:
            head = (f"`{s.command}` "
                    + ("TIMED OUT" if s.timed_out else f"exited {s.exit_code}"))
            out.append(f"\n[{s.name}] {head}\n{s.output_tail}".rstrip())
        return "\n".join(out)


def _tail(text: str, lines: int = _TAIL_LINES) -> str:
    rows = (text or "").splitlines()
    return "\n".join(rows[-lines:]).strip()


def _looks_missing(exit_code: int, out: str, err: str) -> bool:
    if exit_code in _NOT_FOUND_CODES:
        return True
    blob = (err + "\n" + out).lower()
    return any(h in blob for h in _NOT_FOUND_HINTS)


async def verify_workspace(
    root: str,
    *,
    steps: tuple[str, ...] = ("build", "test"),
    install: bool = False,
    timeout: int = 240,
) -> VerifyReport:
    """Detect the build system and run the requested steps.

    `steps` ⊆ {build, test, lint}. `install` runs the dependency-install command
    first (off by default — it needs the network and is slow). A missing
    toolchain is reported as `skipped`, not a failure.
    """
    bs = detect_build_system(root)
    if bs is None:
        return VerifyReport(system=None)

    report = VerifyReport(system=bs.name, language=bs.language)

    async def _run_step(name: str, command: str | None) -> None:
        if not command:
            report.steps.append(
                VerifyStep(name=name, command="(none)", skipped=True))
            return
        r = await run_in_workspace(command, cwd=root, timeout=timeout)
        if r.denied:
            report.steps.append(VerifyStep(
                name=name, command=command, exit_code=r.exit_code,
                ok=False, skipped=True,
                output_tail=f"(skipped: {r.reason})"))
            return
        if _looks_missing(r.exit_code, r.stdout, r.stderr) and not r.timed_out:
            report.steps.append(VerifyStep(
                name=name, command=command, exit_code=r.exit_code,
                skipped=True, output_tail="(toolchain not installed)"))
            return
        report.steps.append(VerifyStep(
            name=name, command=command, exit_code=r.exit_code,
            ok=r.ok, timed_out=r.timed_out,
            output_tail=_tail(r.stdout + ("\n" + r.stderr if r.stderr else "")),
        ))

    if install and bs.install:
        await _run_step("install", bs.install)
        # If install failed hard, still try build/test — they'll surface it.

    cmd_for = {"build": bs.build, "test": bs.test, "lint": bs.lint}
    for name in steps:
        if name in cmd_for:
            await _run_step(name, cmd_for[name])

    return report


__all__ = ["VerifyReport", "VerifyStep", "verify_workspace"]
