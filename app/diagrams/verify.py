"""Verify a Mermaid diagram before it is ever rendered.

The problem
-----------
When a user pastes Mermaid into a prompt, the old path handed it straight to the
webview. If it was malformed the user watched a render fail, and only then did
the repair loop start. That is the wrong order: the diagram should be understood,
checked and fixed *before* it reaches the screen.

The three stages
----------------
1. **Static validation** — `app/diagrams/validators.validate_source`, which
   already knows the grammar: unbalanced `subgraph`/`end`, unquoted labels with
   parentheses, single-dash arrows, undeclared nodes.
2. **Sandbox execution** — the checks a static pass cannot make honestly are run
   as a real program, in the sandbox, on the actual source. It is genuine
   execution against the user's input, and it runs with the same isolation as
   any other generated code (`network_mode: none`, dropped caps, a pids cap).
3. **LLM repair, then re-verify** — a failure carries the exact diagnostic back
   into a syntax-only repair, and the result is verified again. Bounded, because
   a model that cannot fix a diagram in two attempts will not fix it in ten.

Why the sandbox rather than importing mermaid here
--------------------------------------------------
Mermaid's own parser is a browser bundle: `mermaid.parse` wants a DOM, and the
sandbox has no network to install `jsdom` or `mermaid-cli`. So the sandbox stage
runs a **self-contained structural checker** — one that walks the source the way
a parser does (statement by statement, tracking block depth and quoting state)
rather than pattern-matching lines. It catches the failures that actually reach
users, and it does so by executing code on the input rather than by asserting.

The authoritative parser is still the FE webview; this stage exists so the user
does not have to see it fail first. Everything fails **open**: a sandbox that is
unavailable degrades to static validation, never to a blocked diagram.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger("zapthetrick.diagrams.verify")

# Repair attempts before giving up and returning the best source we have. A
# model that cannot fix a diagram in two passes will not fix it in ten, and each
# pass costs a round-trip the user is waiting on.
MAX_REPAIRS = 2
# The sandbox check is a few hundred lines of JS over a small string; anything
# slower than this means the sandbox is wedged, not that the diagram is hard.
VERIFY_TIMEOUT_S = 20.0

# The structural checker, run inside the sandbox against the diagram source.
# Written as a real scanner rather than a set of regexes because the failures
# that matter are STATEFUL: whether a bracket is inside quotes, whether a
# subgraph was closed, whether an arrow has a head. A line-at-a-time regex
# cannot answer those.
_CHECKER_JS = r"""
'use strict';
// The diagram arrives on STDIN, not argv: a diagram is multi-line and can
// contain quotes, backticks and shell metacharacters, and passing it as an
// argument would make correctness depend on quoting rules.
const source = require('fs').readFileSync(0, 'utf8');
const errors = [];
const warnings = [];

const DIAGRAM_KINDS = [
  'graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram',
  'stateDiagram-v2', 'erDiagram', 'journey', 'gantt', 'pie', 'quadrantChart',
  'requirementDiagram', 'gitGraph', 'mindmap', 'timeline', 'sankey-beta',
  'xychart-beta', 'block-beta', 'C4Context', 'C4Container', 'zenuml',
];

const lines = source.split(/\r?\n/);
let header = null;
let headerLine = 0;
for (let i = 0; i < lines.length; i++) {
  const t = lines[i].trim();
  if (!t || t.startsWith('%%')) continue;          // blank or comment
  header = t;
  headerLine = i + 1;
  break;
}

if (header === null) {
  errors.push({ line: 1, message: 'the diagram is empty' });
} else {
  const kind = DIAGRAM_KINDS.find(
    (k) => header === k || header.startsWith(k + ' ') || header.startsWith(k + '\t'));
  if (!kind) {
    errors.push({
      line: headerLine,
      message: 'first statement does not declare a diagram type (expected one of: '
        + DIAGRAM_KINDS.slice(0, 8).join(', ') + ', ...)',
    });
  } else if (kind === 'graph' || kind === 'flowchart') {
    const dir = header.slice(kind.length).trim().split(/\s+/)[0] || '';
    if (dir && !/^(TB|TD|BT|RL|LR)$/.test(dir)) {
      errors.push({ line: headerLine, message: 'invalid direction "' + dir + '" (expected TB, TD, BT, RL or LR)' });
    }
  }
}

// Stateful scan: quoting, bracket balance and block depth, tracked the way a
// parser tracks them rather than matched per line.
let depth = 0;
const openBlocks = [];
for (let i = 0; i < lines.length; i++) {
  const raw = lines[i];
  const lineNo = i + 1;
  const trimmed = raw.trim();
  if (!trimmed || trimmed.startsWith('%%')) continue;

  let inQuote = false;
  let square = 0, round = 0, curly = 0;
  for (let c = 0; c < raw.length; c++) {
    const ch = raw[c];
    if (ch === '"' && raw[c - 1] !== '\\') { inQuote = !inQuote; continue; }
    if (inQuote) continue;
    if (ch === '[') square++;
    else if (ch === ']') square--;
    else if (ch === '(') round++;
    else if (ch === ')') round--;
    else if (ch === '{') curly++;
    else if (ch === '}') curly--;
    if (square < 0 || round < 0 || curly < 0) {
      errors.push({ line: lineNo, message: 'unbalanced bracket — a closing bracket with no opener' });
      break;
    }
  }
  if (inQuote) errors.push({ line: lineNo, message: 'unterminated double quote' });
  if (square > 0) errors.push({ line: lineNo, message: 'unclosed "[" — node label is not terminated' });
  if (round > 0) errors.push({ line: lineNo, message: 'unclosed "(" — wrap a label containing parentheses in double quotes' });
  if (curly > 0) errors.push({ line: lineNo, message: 'unclosed "{"' });

  // Parentheses inside an UNQUOTED node label. The brackets balance, so a
  // balance-only check waves this through — and then mermaid fails to render
  // it. The fix mermaid wants is quoting: A["Fetch (REST)"].
  // Strip the SHAPE delimiters first — `[(cylinder)]`, `([stadium])`,
  // `[[subroutine]]`, `((circle))`, `{{hexagon}}` are all valid mermaid, and
  // flagging them would send the repair loop chasing correct syntax.
  const deshaped = raw
    .replace(/\[\(([^)]*)\)\]/g, '[$1]')
    .replace(/\(\[([^\]]*)\]\)/g, '[$1]')
    .replace(/\[\[([^\]]*)\]\]/g, '[$1]')
    .replace(/\(\(([^)]*)\)\)/g, '[$1]')
    .replace(/\{\{([^}]*)\}\}/g, '[$1]');
  const labelRe = /\[([^\]]*)\]/g;
  let lm;
  while ((lm = labelRe.exec(deshaped)) !== null) {
    const label = lm[1];
    if (!label) continue;
    const quoted = label.trim().startsWith('"') && label.trim().endsWith('"');
    if (!quoted && /[()]/.test(label)) {
      errors.push({
        line: lineNo,
        message: 'unclosed "(" in an unquoted label — wrap it in double quotes '
          + '(A["Fetch (REST)"])',
      });
    }
  }

  const word = trimmed.split(/\s+/)[0];
  if (word === 'subgraph') { depth++; openBlocks.push(lineNo); }
  else if (word === 'end') {
    depth--;
    openBlocks.pop();
    if (depth < 0) {
      errors.push({ line: lineNo, message: '"end" with no matching "subgraph"' });
      depth = 0;
    }
  }

  // A single-dash link is the single most common hand-written mistake.
  if (/(^|\s)-(>|-[^->])/.test(trimmed) && !/-->|---|-\.-|==>/.test(trimmed)) {
    if (/\s->\s/.test(trimmed)) {
      errors.push({ line: lineNo, message: 'link needs at least two dashes ("-->" not "->")' });
    }
  }
  // A bare "A -- B" is a link with no terminator.
  if (/\s--\s+[A-Za-z0-9_"]/.test(trimmed) && !/-->|---|--[>ox]|--\s*\|/.test(trimmed)) {
    warnings.push({ line: lineNo, message: 'link has no arrowhead or terminator ("A --- B" or "A --> B")' });
  }
}

for (const ln of openBlocks) {
  errors.push({ line: ln, message: 'subgraph opened here is never closed with "end"' });
}

process.stdout.write(JSON.stringify({
  ok: errors.length === 0,
  errors: errors.slice(0, 20),
  warnings: warnings.slice(0, 20),
}));
"""


@dataclass
class VerifyReport:
    """What happened to a diagram on its way to the screen."""

    ok: bool
    source: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repairs: int = 0
    # Which stages actually ran. A stage that could not run is not a pass —
    # saying so is the difference between "verified" and "not checked".
    stages: list[str] = field(default_factory=list)
    sandbox_available: bool = True

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "errors": self.errors, "warnings": self.warnings,
            "repairs": self.repairs, "stages": self.stages,
            "sandbox_available": self.sandbox_available,
        }


def _static_errors(source: str) -> tuple[list[str], list[str]]:
    """Stage 1 — the existing deterministic validators."""
    try:
        from app.diagrams.validators import validate_source
        report = validate_source(source)
        errs, warns = [], []
        for f in getattr(report, "findings", []) or []:
            msg = getattr(f, "message", "") or str(f)
            sev = (getattr(f, "severity", "") or "").lower()
            cat = (getattr(f, "category", "") or "").lower()
            # Only SYNTAX blocks. Style and accessibility findings are real and
            # worth surfacing, but they do not stop a diagram rendering — and a
            # syntax-only repair cannot fix them, so treating them as errors
            # would burn every retry on something the model must not touch.
            blocking = sev in ("error", "critical") and cat in ("syntax", "")
            (errs if blocking else warns).append(msg)
        return errs, warns
    except Exception:  # noqa: BLE001 — validation must never block a diagram
        log.debug("static mermaid validation unavailable", exc_info=True)
        return [], []


def sandbox_check(source: str) -> tuple[bool, list[str], list[str], bool]:
    """Stage 2 — run the structural checker on the source, in the sandbox.

    Returns `(ok, errors, warnings, available)`. `available` is False when the
    sandbox could not run at all, which is reported rather than silently
    counted as a pass.
    """
    try:
        from app.sandbox.executor import SandboxLimits, run_code
        res = run_code(
            _CHECKER_JS, "javascript", stdin=source,
            limits=SandboxLimits(timeout_s=VERIFY_TIMEOUT_S),
        )
    except Exception:  # noqa: BLE001 — no sandbox ⇒ "not checked", not "passed"
        return True, [], [], False

    if getattr(res, "status", "") == "unavailable" or not getattr(res, "stdout", ""):
        return True, [], [], False
    try:
        payload = json.loads((res.stdout or "").strip())
    except Exception:  # noqa: BLE001 — a garbled result is "not checked"
        return True, [], [], False

    def _fmt(items):
        return [f"line {i.get('line', '?')}: {i.get('message', '')}"
                for i in (items or [])]

    return (bool(payload.get("ok")), _fmt(payload.get("errors")),
            _fmt(payload.get("warnings")), True)


async def _repair(source: str, error: str) -> str:
    """Stage 3 — syntax-only LLM repair, reusing the existing prompt so the
    repair contract is identical to the FE's compile/repair loop."""
    try:
        from app.diagrams.repair_contract import (_REPAIR_PROMPT,
                                                  _strip_fences)
        from app.core.llm_client import LLMClient
        prompt = _REPAIR_PROMPT.format(error=error[:2000], source=source[:20000])
        text = await LLMClient().complete([{"role": "user", "content": prompt}])
        fixed = _strip_fences(text or "")
        return fixed or source
    except Exception:  # noqa: BLE001 — a failed repair returns the original
        log.debug("mermaid repair failed", exc_info=True)
        return source


async def verify(source: str, *, repair: bool = True,
                 max_repairs: int = MAX_REPAIRS) -> VerifyReport:
    """Validate → verify in the sandbox → repair → re-verify.

    Returns the best source it achieved plus an honest report. A diagram that
    cannot be fixed is returned UNCHANGED with its errors rather than replaced
    by something invented — a wrong diagram is worse than a broken one, because
    the user cannot tell it is wrong.
    """
    src = (source or "").strip()
    report = VerifyReport(ok=False, source=src)
    if not src:
        report.errors = ["the diagram is empty"]
        return report

    for attempt in range(max_repairs + 1):
        stages = ["static"]
        errs, warns = _static_errors(src)

        ok_sandbox, sb_errs, sb_warns, available = sandbox_check(src)
        if available:
            stages.append("sandbox")
        errs = errs + sb_errs
        warns = warns + sb_warns

        report = VerifyReport(
            ok=not errs, source=src, errors=errs, warnings=warns,
            repairs=attempt, stages=stages, sandbox_available=available)

        if report.ok or not repair or attempt >= max_repairs:
            return report

        fixed = await _repair(src, "\n".join(errs[:6]))
        if fixed.strip() == src.strip():
            return report          # the model changed nothing — stop burning calls
        src = fixed.strip()

    return report


__all__ = ["VerifyReport", "verify", "sandbox_check", "MAX_REPAIRS"]
