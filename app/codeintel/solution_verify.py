"""Sandbox-verify a generated coding solution against the problem's examples.

For an image coding problem (LeetCode/HackerRank), after the model writes the
solution we COMPILE + RUN it against the visible Input/Output examples in the
sandbox, so "optimal solution" is *verified*, not merely asserted.

Approach — an LLM-built self-contained harness:
  A `class Solution { method }` stub isn't runnable on its own, and building a
  driver for an arbitrary signature/type across languages by hand is brittle.
  Instead we ask the model to emit ONE self-contained program (the solution +
  a main that runs each example and prints `CASE <i> PASS|FAIL got=.. want=..`),
  then run THAT in the sandbox and read the ground-truth PASS/FAIL. The runtime
  is the judge, not the model's prose.

Only runs when a runtime for the language is actually installed (python/node/
java out of the box; rust/cpp/go when their toolchain is present) — otherwise it
reports `skipped` with an honest reason. Everything is fail-open.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

def runtime_for(language: str) -> tuple[str, str] | None:
    """(sandbox_lang_id, tool) for a language label via the sandbox registry,
    or None if the language isn't in the registry."""
    from app.sandbox import lang_registry
    cid = lang_registry.canonical(language)
    if cid is None:
        return None
    tool = lang_registry.check_tool(cid)
    return (cid, tool) if tool else None


def runtime_available(language: str) -> bool:
    """True when the language can actually be run by the active sandbox backend:
    the docker container (docker backend) or the host toolchain (local)."""
    from app.sandbox import lang_registry
    cid = lang_registry.canonical(language)
    if not cid:
        return False
    try:
        from app.core.config_loader import cfg
        if str(getattr(cfg.sandbox, "backend", "local") or "local").lower() \
                == "docker":
            from app.sandbox import docker_exec
            return lang_registry.container_supports(cid) and docker_exec.available()
    except Exception:  # noqa: BLE001
        pass
    return lang_registry.is_available(cid)


def extract_examples(text: str, limit: int = 6) -> list[dict]:
    """Pull Input/Output example pairs out of the OCR'd problem text."""
    if not text:
        return []
    out: list[dict] = []
    # "[Sample] Input: <...> [Sample] Output: <...>" — tolerant of newlines and
    # the OCR's spacing. CRITICAL: skip the "Input Format" / "Output Format"
    # SECTION HEADERS (the `(?!\s+Format\b)` guards) — matching those grabbed the
    # prose "On the first line, print US: u where…" as the expected output and
    # failed every correct solution (the HackerRank currency-formatter bug). The
    # optional `Sample ` prefix is consumed as part of the marker so the number
    # isn't polluted with a trailing "Sample".
    for m in re.finditer(
            r"(?:Sample\s+)?Input(?!\s+Format\b)\s*:?\s*(.+?)"
            r"\s*(?:Sample\s+)?Output(?!\s+Format\b)\s*:?\s*(.+?)"
            r"(?=\n\s*(?:Explanation|Example|Sample|Input|Constraints|Note)\b|\Z)",
            text, re.IGNORECASE | re.DOTALL):
        inp = " ".join(m.group(1).split())[:400]
        # Raw (newline-preserving) input for stdin-style problems, where the
        # Input block IS the program's standard input.
        inp_raw = m.group(1).strip()[:600]
        out_v = " ".join(m.group(2).split())[:200]
        if inp and out_v:
            out.append({"input": inp, "input_raw": inp_raw, "expected": out_v})
        if len(out) >= limit:
            break
    return out


_STDIN_RE = re.compile(
    r"input\s*\(|raw_input\s*\(|sys\.stdin|std\s*::\s*cin|\bcin\b|scanf|getchar|"
    r"\bgets\b|fgets|getline|Scanner\b|System\.in|BufferedReader|readLine|"
    r"process\.stdin|require\(['\"]readline|bufio\.New|fmt\.Scan|os\.Stdin|"
    r"io::stdin|read_line|\$stdin|STDIN|<STDIN>|Console\.ReadLine|\breadln\b",
    re.IGNORECASE)


def _reads_stdin(code: str) -> bool:
    """True when the solution reads its input from STDIN (competitive/Codeforces/
    HackerRank style) rather than exposing a function to call (LeetCode style)."""
    return bool(code and _STDIN_RE.search(code))


def _norm_out(s: str) -> str:
    """Normalize program output for a tolerant compare: trim, collapse internal
    whitespace, and peel surrounding quotes ("6" vs 6, trailing newline, etc.)."""
    s = re.sub(r"\s+", " ", (s or "").strip())
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _outputs_match(got: str, expected: str) -> bool:
    """Tolerant output equality — ignores whitespace, surrounding quotes, and
    list punctuation ("[1, 2]" == "[1,2]" == "1 2"). Catches the false negatives
    where a CORRECT answer differs from the example only by formatting."""
    g, e = _norm_out(got), _norm_out(expected)
    if g == e:
        return True
    _strip = lambda x: re.sub(r"[\s,]+", "", x)   # noqa: E731
    return _strip(g) == _strip(e)


def _parse_struct(s: str):
    """Best-effort parse of a printed result into a Python structure. Handles the
    common list/tuple/number/bool/string shapes most languages print (Elixir,
    Python, JS, … all serialize a list of int-lists as `[[1, 1, 2], …]`). Returns
    None when it can't parse — the caller then falls back to string compare."""
    import ast
    t = (s or "").strip()
    if not t:
        return None
    # Normalize a few non-Python spellings before literal_eval.
    t2 = (t.replace("true", "True").replace("false", "False")
          .replace("null", "None").replace("nil", "None"))
    try:
        return ast.literal_eval(t2)
    except Exception:  # noqa: BLE001
        return None


def _canon(x):
    """Canonical, hashable, order-normalizing key for a value so two structurally
    equal results compare equal regardless of surface formatting."""
    if isinstance(x, (list, tuple)):
        return tuple(_canon(v) for v in x)
    return x


def _verdict(got: str, want: str) -> bool:
    """Deterministic PASS/FAIL for one example — computed in PYTHON, never by the
    (LLM-written) harness, so a broken solution can't self-certify. An EMPTY or
    partial result can NEVER match a non-empty expected.

    Accepts an unordered match for a COLLECTION-OF-RESULTS output (the classic
    'return all X in any order' problem) via a multiset compare of the outer
    elements, while a flat/scalar output must match tolerantly (order kept)."""
    g, w = (got or "").strip(), (want or "").strip()
    # 1) exact / tolerant string match (fast path, order-sensitive).
    if _outputs_match(g, w):
        return True
    gp, wp = _parse_struct(g), _parse_struct(w)
    if gp is None or wp is None:
        return False
    # 2) list-of-lists ("collection of results in any order"): the OUTER order is
    #    free, so compare as a MULTISET of canonicalized inner elements. Empty vs
    #    non-empty → different multiset → FAIL (this is the fix).
    if (isinstance(wp, list) and isinstance(gp, list) and wp
            and all(isinstance(v, (list, tuple)) for v in wp)):
        return sorted(map(repr, map(_canon, gp))) == \
            sorted(map(repr, map(_canon, wp)))
    # 3) everything else: structural equality (order kept).
    return _canon(gp) == _canon(wp)


async def _generate_examples(problem_text: str, language_label: str,
                             on_stage=None, n: int = 3) -> list[dict]:
    """When the problem shows NO worked examples, ask the model to INVENT a few
    small, unambiguous test cases so verification can still run. Best-effort;
    returns [] on any trouble. The verdict flags these as generated."""
    try:
        await _emit(on_stage, "Deriving test cases from the problem")
        import json
        prompt = (
            f"Problem:\n{problem_text[:2000]}\n\n"
            f"This problem shows NO worked examples. Invent {n} small test cases "
            "you are CONFIDENT are correct and unambiguous (avoid problems with "
            "multiple valid answers). For a function-style problem, `input` is the "
            "argument(s); for a stdin-style problem, `input` is the raw standard "
            "input. Output ONLY a JSON array of "
            '{"input": <string>, "expected": <string>} — nothing else.')
        txt = await _stream_complete(
            [{"role": "user", "content": prompt}], "standard",
            session_key="verify-examples")
        m = re.search(r"\[.*\]", txt or "", re.DOTALL)
        if not m:
            return []
        out = []
        for d in (json.loads(m.group(0)) or [])[:6]:
            if isinstance(d, dict) and "input" in d and "expected" in d:
                s_in, s_exp = str(d["input"]), str(d["expected"])
                out.append({"input": s_in, "input_raw": s_in, "expected": s_exp})
        return out
    except Exception:  # noqa: BLE001
        return []


def _extract_code_block(text: str) -> str:
    """The (last, usually most complete) fenced code block in the answer."""
    blocks = re.findall(r"```[a-zA-Z0-9+#]*\n(.*?)```", text or "", re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return ""


_HARNESS_SYS = (
    "You turn a coding-problem solution into a single self-contained, runnable "
    "program that TESTS it against the given examples. Output ONLY code in one "
    "fenced block — no prose."
)


# How the sandbox actually RUNS each language — the entry-point shape a harness
# MUST take or it won't even compile. Keyed by canonical id; only languages whose
# entry point is non-obvious need an entry (the rest use the obvious main()).
_HARNESS_ENTRY: dict[str, str] = {
    "erlang": (
        "This is run with `escript main.erl`, so the WHOLE program must be a "
        "module named `main` that exports `main/1`. Structure it EXACTLY like:\n"
        "```erlang\n-module(main).\n-export([main/1]).\n\n"
        "%% paste the candidate solution's functions here (multiply/2 + helpers)\n\n"
        "main(_) ->\n"
        "    %% for each example: Got = multiply(...), then\n"
        "    io:format(\"CASE 1 PASS~n\"),  %% or CASE 1 FAIL got=.. want=..\n"
        "    ok.\n```\n"
        "Do NOT use a different module name and do NOT define main/0. Print with "
        "io:format. Binaries print with ~s, integers/terms with ~p."),
    "elixir": (
        "Run with `elixir main.exs` (a script). Put the solution module + a "
        "top-level loop that prints each CASE line with IO.puts. No `defmodule "
        "Main do ... def main` wrapper is needed — top-level code runs."),
    "haskell": (
        "Run with `runghc main.hs` — it needs `main :: IO ()`. Call the solution "
        "from main and print each CASE line with putStrLn."),
    "racket": (
        "Run with `racket main.rkt`. Start with `#lang racket`. Top-level "
        "expressions run; print each CASE line with (displayln ...)."),
    "bash": (
        "Run with `bash main.sh`. Use echo for each CASE line."),
    "sql": (
        "Run with `sqlite3`. Emit each CASE line via SELECT (e.g. "
        "`SELECT 'CASE 1 PASS';`)."),
}


def _harness_prompt(problem: str, solution: str, lang_label: str,
                    examples: list[dict]) -> str:
    from app.sandbox import lang_registry
    cid = lang_registry.canonical(lang_label) or ""
    ex = "\n".join(
        f"  Example {i + 1}: input = {e['input']} ; expected output = {e['expected']}"
        for i, e in enumerate(examples))
    entry = _HARNESS_ENTRY.get(cid, "")
    entry_block = f"\nHOW THIS LANGUAGE IS RUN (obey exactly):\n{entry}\n" if entry else ""
    return (
        f"Language: {lang_label}\n\n"
        f"Problem (verbatim from the source):\n{problem[:2000]}\n\n"
        f"Candidate solution:\n```\n{solution[:4000]}\n```\n\n"
        f"Examples to check:\n{ex}\n"
        f"{entry_block}\n"
        "Write ONE complete, self-contained program in the SAME language that:\n"
        "1. includes the candidate solution EXACTLY as given (same class/method "
        "names/signature);\n"
        "2. in its entry point, runs EACH example: parse the input the same way "
        "the problem states it, call the solution, and capture its ACTUAL "
        "returned result;\n"
        "3. prints EXACTLY one line per example, nothing else per case:\n"
        "   `CASE <n> got=<actual result> want=<expected output>`\n"
        "   — where <actual result> is the solution's real return value serialized "
        "COMPACTLY on ONE line (square brackets, comma-separated, e.g. "
        "`[[1,1,2],[1,2,1],[2,1,1]]`) and <expected output> is the example's "
        "expected value serialized the same way.\n"
        "   DO NOT decide pass/fail and DO NOT compare the values yourself — just "
        "PRINT the real `got` and the `want`. The judge compares them.\n"
        "Use 1-based <n>. The program MUST compile and run as-is. Do not print "
        "anything else. Output ONLY the program in a single fenced code block."
    )


_CASE_RE = re.compile(r"^CASE\s+(\d+)\s+(PASS|FAIL)\b(.*)$", re.MULTILINE)
# The RUNNER format: the harness prints the raw result; PYTHON judges (so a
# broken solution can't self-certify by writing a lenient comparison).
_GOTWANT_RE = re.compile(r"^CASE\s+(\d+)\s+got=(.*?)\s+want=(.*)$", re.MULTILINE)


def _cases_from_run(stdout: str):
    """Parse harness output into (n, verdict, detail) tuples. Prefers the
    got=/want= runner format (Python computes the verdict via `_verdict`); falls
    back to a self-judged CASE PASS/FAIL line only if the harness ignored the
    runner instruction."""
    gw = _GOTWANT_RE.findall(stdout or "")
    if gw:
        out = []
        for _n, _got, _want in gw:
            _ok = _verdict(_got, _want)
            out.append((_n, "PASS" if _ok else "FAIL",
                        f" got={_got.strip()[:60]} want={_want.strip()[:60]}"))
        return out
    return _CASE_RE.findall(stdout or "")


async def _emit(on_stage, name: str) -> None:
    """Best-effort progress ping (drives the live 'stage' chips). Never raises."""
    if on_stage is None:
        return
    try:
        await on_stage(name)
    except Exception:  # noqa: BLE001
        pass


def _finalize_cases(result: dict, cases: list, examples: list) -> None:
    """Fold parsed CASE verdicts into the result dict. 'Passed' = every case
    that ran passed (a harness may test more/fewer than we extracted, so
    comparing to len(examples) would mislabel a green run)."""
    passed = sum(1 for _n, verdict, _rest in cases if verdict == "PASS")
    fails = [f"case {n}:{rest.strip()}" for n, verdict, rest in cases
             if verdict == "FAIL"]
    result["total"] = len(cases) or len(examples)
    result["passed"] = passed
    result["details"] = fails[:6]
    result["status"] = "passed" if passed == len(cases) else "failed"


# ── Deterministic Python harness ────────────────────────────────────────────
# For the overwhelmingly common `class Solution: def method(self, …)` LeetCode
# shape in Python we build the test driver DETERMINISTICALLY (parse the method +
# `ast.literal_eval` the example inputs) — no LLM call, so it's instant and can't
# hallucinate a wrong harness. Anything it isn't sure about (custom types like
# TreeNode/ListNode, multiple public methods, unparseable inputs) returns None
# and the caller falls back to the LLM harness.

def _split_top_commas(s: str) -> list[str]:
    """Split on commas that are NOT inside brackets/quotes (so `nums = [1,2],
    target = 3` → ['nums = [1,2]', 'target = 3'])."""
    parts, depth, cur, q = [], 0, [], None
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_example_input(inp: str):
    """Parse a LeetCode-style input line into ('kw', {name: value}) or
    ('pos', [values]) via ast.literal_eval. None when it can't be parsed
    unambiguously (→ fall back to the LLM harness)."""
    import ast
    pieces = _split_top_commas(inp or "")
    if not pieces:
        return None
    kw, pos = {}, []
    for p in pieces:
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", p, re.DOTALL)
        try:
            if m:
                kw[m.group(1)] = ast.literal_eval(m.group(2))
            else:
                pos.append(ast.literal_eval(p))
        except Exception:  # noqa: BLE001 — non-literal input → give up (LLM path)
            return None
    if kw and pos:
        return None                      # mixed named/positional → ambiguous
    return ("kw", kw) if kw else ("pos", pos)


def _python_harness(solution: str, examples: list[dict]) -> str | None:
    """A complete, self-contained Python test program (candidate + driver that
    prints `CASE n got= want=`) built WITHOUT an LLM, or None when the shape is
    outside the deterministic fast-path."""
    import ast
    if not solution or "class Solution" not in solution:
        return None
    # Custom linked/tree types need construction the LLM harness does better.
    if any(t in solution for t in ("TreeNode", "ListNode", "class Node")):
        return None
    try:
        tree = ast.parse(solution)
    except Exception:  # noqa: BLE001
        return None
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Solution"), None)
    if cls is None:
        return None
    methods = [m for m in cls.body
               if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")]
    if len(methods) != 1:                # ambiguous target → LLM harness
        return None
    meth = methods[0]
    params = [a.arg for a in meth.args.args if a.arg != "self"]
    if not params:
        return None

    rows = []
    for ex in examples:
        parsed = _parse_example_input(ex.get("input") or "")
        if parsed is None:
            return None
        kind, vals = parsed
        if kind == "pos":
            if len(vals) != len(params):
                return None
            args = vals
        else:
            if set(vals) == set(params):
                args = [vals[p] for p in params]     # match by name
            elif len(vals) == len(params):
                args = list(vals.values())           # else by input order
            else:
                return None
        want = " ".join((ex.get("expected") or "").split())
        rows.append(", ".join(repr(a) for a in args) + ",")
        rows[-1] = f"    (({rows[-1]}), {want!r}),"
    if not rows:
        return None

    body = [
        solution.rstrip(), "",
        "import json, copy",
        "def _c(x):",
        "    try:",
        "        return json.dumps(x, separators=(',', ':'))",
        "    except Exception:",
        "        return repr(x)",
        "_S = Solution()",
        f"_M = _S.{meth.name}",
        "_cases = [",
        *rows,
        "]",
        "for _i, (_args, _want) in enumerate(_cases, 1):",
        "    _a = [copy.deepcopy(_x) for _x in _args]",
        "    _got = _M(*_a)",
        "    if _got is None and _a:",     # in-place-mutation problems (e.g. rotate)
        "        _got = _a[0]",
        "    print('CASE %d got=%s want=%s' % (_i, _c(_got), _want))",
    ]
    return "\n".join(body)


_TIER_ORDER = ("trivial", "standard", "hard", "expert")


async def _stream_complete(msgs: list[dict], difficulty: str,
                           session_key: str | None = None) -> str:
    """Get a one-shot completion via the STREAMING path (`llm.stream_chat`),
    collected into a single string.

    Why not `complete_routed`? That is the non-streaming `route_and_complete`
    path: it picks ONE route with NO cross-provider failover, so the instant the
    free models are momentarily busy (e.g. right after the answer consumed a
    slot) it raises `NoRouteAvailable` → 'every model was rate-limited while
    building the test harness' — the exact bug the user hit, even though the
    answer itself streamed fine seconds earlier. `stream_chat` →
    `stream_with_continuation` has the same cross-provider failover the answer
    used, so it finds a route where the one-shot call gives up. Raises `LLMError`
    only when the streaming path itself has no route at all."""
    from app.core.llm_client import llm
    parts: list[str] = []
    async for chunk in llm.stream_chat(
            msgs, session_key=session_key, options={"difficulty": difficulty}):
        parts.append(chunk)
    return "".join(parts)


def _harness_tiers(seed: str | None) -> tuple[str, str, str]:
    """Escalating model tiers for the harness build. Seed with the difficulty the
    ANSWER was generated at (that model pool JUST succeeded), so we don't burn the
    first attempts on weaker/free tiers that are the first to be rate-limited — the
    exact cause of 'every model was rate-limited while building the test harness'
    on a solution that itself generated fine. Always climbs toward expert."""
    base = (seed or "standard").lower()
    if base not in _TIER_ORDER or base == "trivial":
        base = "standard"
    i = _TIER_ORDER.index(base)
    nxt = min(i + 1, len(_TIER_ORDER) - 1)
    return (_TIER_ORDER[i], _TIER_ORDER[nxt], "expert")


async def verify_solution(problem_text: str, answer_text: str,
                          language_label: str, examples: list[dict] | None = None,
                          on_stage=None, min_difficulty: str | None = None):
    """Compile+run the answer's solution against `examples` in the sandbox.

    Returns a dict: {status, passed, total, details, reason}. `status` is one of
    passed | failed | error | skipped. `on_stage(name)` (optional async cb) is
    pinged at each step so the caller can stream progress. `min_difficulty` seeds
    the harness-build model tier (pass the difficulty the ANSWER used, so the
    harness routes to the same capable pool). Never raises."""
    result = {"status": "skipped", "passed": 0, "total": 0,
              "details": [], "reason": ""}
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.sandbox, "enabled", True)):
            result["reason"] = "sandbox disabled"
            return result

        _generated = False
        examples = examples or extract_examples(problem_text)
        if not examples:
            # No worked examples in the problem → derive a few so we can still
            # verify (flagged as generated in the verdict).
            examples = await _generate_examples(
                problem_text, language_label, on_stage=on_stage)
            _generated = bool(examples)
        if not examples:
            result["reason"] = "no examples found or derivable from the problem"
            return result

        rt = runtime_for(language_label)
        if rt is None:
            result["reason"] = f"unsupported language: {language_label}"
            return result
        sandbox_lang, tool = rt
        if not runtime_available(language_label):
            result["reason"] = (
                f"{language_label} runtime not installed (needs '{tool}' on "
                f"PATH); install it to enable verification")
            return result

        solution = _extract_code_block(answer_text)
        if not solution:
            result["reason"] = "no code block in the answer"
            return result
        result["generated"] = _generated

        import asyncio
        from app.sandbox.executor import run_batch, run_code
        from app.sandbox import lang_registry as _lr
        _ver = _lr.version_from_label(language_label)   # e.g. Python2 → 2.7
        _n = len(examples)
        _cases = f"{_n} test case{'s' if _n != 1 else ''}"

        # ── stdin-style (competitive) problems ─────────────────────────────
        # The solution reads STDIN and prints the answer. Run it DIRECTLY per
        # example (feeding the input on stdin) and compare stdout — no LLM
        # harness needed, so it's immune to harness-misparse false negatives.
        if _reads_stdin(solution):
            await _emit(on_stage,
                        f"Running {language_label} against {_cases} (stdin)")
            passed, fails = 0, []
            result["total"] = _n
            # Compile ONCE, run the artifact against every example's stdin (was:
            # a full compile+run per example — N× the docker overhead).
            _inputs = []
            for ex in examples:
                _in = (ex.get("input_raw") or ex.get("input") or "")
                if not _in.endswith("\n"):
                    _in += "\n"
                _inputs.append(_in)
            runs = await asyncio.to_thread(
                run_batch, solution, sandbox_lang, version=_ver, stdins=_inputs)
            for i, (ex, run) in enumerate(zip(examples, runs), 1):
                if run.status == "timeout":
                    result["status"] = "error"
                    result["reason"] = "the solution timed out"
                    return result
                if run.status != "ok" and not (run.stdout or "").strip():
                    result["status"] = "error"
                    result["reason"] = (run.stderr or run.reason
                                        or "run failed")[:400]
                    return result
                # Use the structural verdict (order-aware + set-aware for
                # collection outputs; an empty result never matches a non-empty
                # expected) — same judge as the harness path.
                if _verdict(run.stdout or "", ex["expected"]):
                    passed += 1
                else:
                    fails.append(f"case {i}: got={_norm_out(run.stdout or '')[:40]} "
                                 f"want={_norm_out(ex['expected'])[:40]}")
            result["passed"] = passed
            result["details"] = fails[:6]
            result["status"] = "passed" if passed == _n else "failed"
            return result

        # ── function-style (LeetCode) problems ─────────────────────────────
        # FAST PATH: for the common Python `class Solution` shape, build the test
        # driver DETERMINISTICALLY (no LLM) — instant + can't hallucinate. Only a
        # run that prints CASE verdicts counts; otherwise fall through to the LLM
        # harness (exotic signatures, custom types, other languages).
        if sandbox_lang == "python":
            _prog = _python_harness(solution, examples)
            if _prog:
                await _emit(on_stage, f"Testing {language_label} against {_cases}")
                _drun = await asyncio.to_thread(
                    run_code, _prog, sandbox_lang, version=_ver)
                _dcases = _cases_from_run(_drun.stdout or "")
                if _dcases:
                    _finalize_cases(result, _dcases, examples)
                    return result
                # No verdicts (the candidate itself crashed, or an odd shape) →
                # fall through to the LLM harness rather than failing here.

        # Build a runnable test harness that wraps the solution + calls it. A
        # niche language (Racket, SML, …) can produce a harness that won't
        # COMPILE (a hallucinated identifier). When a run yields NO CASE output,
        # the HARNESS — not the solution — is the likely culprit, so we
        # regenerate the harness (feeding it the compiler error) a few times
        # before giving up. Only a run that PRINTS case verdicts is a real
        # pass/fail signal.
        _harness_feedback = ""
        _last_run = None
        cases: list = []
        for _htry in range(3):
            await _emit(on_stage, (f"Writing the {language_label} test program"
                                   if _htry == 0 else "Rebuilding the test program"))
            _user = _harness_prompt(problem_text, solution, language_label, examples)
            if _harness_feedback:
                _user += (
                    "\n\nYOUR PREVIOUS test program FAILED TO COMPILE/RUN with:\n"
                    f"{_harness_feedback[:600]}\nFix it — use ONLY valid, standard "
                    f"{language_label}, and keep the exact CASE output format.")
            msgs = [{"role": "system", "content": _HARNESS_SYS},
                    {"role": "user", "content": _user}]
            harness_text = None
            _last_exc = None
            # Escalate the model tier per attempt, SEEDED at the difficulty the
            # answer used. The ANSWER was generated at a high difficulty (strong,
            # available models), but starting the harness at "standard" routes to
            # weaker/free models that are the first to be rate-limited or error on
            # the non-streaming harness call — the exact "every model was rate-
            # limited while building the test harness" the user hit on a solution
            # that itself generated fine. Seeding at the answer's tier reuses the
            # same capable pool; exponential backoff survives a brief rate-limit
            # window instead of giving up in ~3s.
            _HARNESS_DIFFS = _harness_tiers(min_difficulty)
            for _attempt in range(3):
                try:
                    # Use the STREAMING route (with cross-provider failover) — the
                    # same path the answer succeeded on — not the failover-less
                    # one-shot complete_routed that was dying with NoRouteAvailable
                    # right after the answer. A fresh session_key per attempt lets
                    # the escalation route to a DIFFERENT model each time.
                    _got = await _stream_complete(
                        msgs, _HARNESS_DIFFS[_attempt],
                        session_key=f"verify-harness:{_htry}:{_attempt}")
                    if (_got or "").strip():
                        harness_text = _got
                        break
                    _last_exc = _last_exc or RuntimeError("empty harness reply")
                except Exception as exc:  # noqa: BLE001 — an LLM outage isn't a verdict
                    _last_exc = exc
                if _attempt < 2:
                    await _emit(on_stage, "Trying a stronger model for the "
                                "test harness")
                    await asyncio.sleep(2.0 * (2 ** _attempt))  # 2s, 4s
            if harness_text is None:
                _msg = str(_last_exc or "")
                _ml = _msg.lower()
                if "route" in _ml or "rate limit" in _ml or "429" in _ml:
                    result["reason"] = (
                        "every model was momentarily rate-limited while building "
                        "the test harness — the solution itself is unchanged")
                else:
                    # Surface the ACTUAL provider error (truncated) — "(LLMError)"
                    # alone is undiagnosable; the real text says what broke.
                    _detail = (_msg.strip()[:160]
                               or (type(_last_exc).__name__ if _last_exc
                                   else "unknown"))
                    result["reason"] = (
                        f"the test-harness step failed ({_detail}) — the "
                        "solution itself is unchanged")
                log.info("verify: harness LLM call failed: %s", _last_exc)
                return result
            program = _extract_code_block(harness_text or "") or (harness_text or "")
            if not program.strip():
                _harness_feedback = "the previous reply contained no code block"
                continue

            _vtag = f" {language_label}" + (f" (v{_ver})" if _ver else "")
            await _emit(on_stage, f"Compiling & executing{_vtag} in the sandbox")
            run = await asyncio.to_thread(
                run_code, program, sandbox_lang, version=_ver)
            _last_run = run
            if run.status == "timeout":
                result["status"] = "error"
                result["reason"] = "the solution timed out"
                return result
            cases = _cases_from_run(run.stdout or "")
            if cases:
                break   # got real verdicts → done
            # No CASE lines → the harness likely didn't compile; retry it with
            # the error as feedback.
            _harness_feedback = (run.stderr or run.reason or "").strip()
            log.info("verify: harness produced no verdicts (try %d/%d): %s",
                     _htry + 1, 3, _harness_feedback[:140])

        await _emit(on_stage, f"Testing against {_cases}")
        if not cases:
            # Every harness attempt failed to produce verdicts (compile error).
            result["status"] = "error"
            result["reason"] = (((_last_run.stderr if _last_run else "") or "")
                                .strip()[:300] or "harness produced no verdicts")
            return result
        _finalize_cases(result, cases, examples)
    except Exception as exc:  # noqa: BLE001 — verification must never break a turn
        log.info("solution_verify failed: %s", exc)
        result["status"] = "skipped"
        result["reason"] = f"verification error ({type(exc).__name__})"
    return result


_FENCE = {
    "python": "python", "java": "java", "cpp": "cpp", "csharp": "csharp",
    "javascript": "javascript", "typescript": "typescript", "go": "go",
    "rust": "rust", "kotlin": "kotlin", "swift": "swift", "ruby": "ruby",
    "php": "php", "bash": "bash", "r": "r", "sql": "sql",
}


def _fence_tag(language_label: str) -> str:
    from app.sandbox import lang_registry
    cid = lang_registry.canonical(language_label) or ""
    return _FENCE.get(cid, cid)


_REPAIR_SYS = (
    "You FIX a coding solution that failed its sandbox tests. Output ONLY the "
    "corrected solution in one fenced code block — keep the same class/method "
    "name and signature, no prose."
)


def _repair_prompt(problem: str, solution: str, lang_label: str,
                   verdict: dict) -> str:
    fails = "\n".join(verdict.get("details", [])) or verdict.get("reason", "")
    return (
        f"Language: {lang_label}\n\nProblem:\n{problem[:1600]}\n\n"
        f"This solution FAILED the sandbox tests:\n```\n{solution[:3000]}\n```\n\n"
        f"Failures reported by the runtime:\n{fails[:800]}\n\n"
        f"Return a CORRECTED, complete solution in {lang_label} that passes "
        f"EVERY example. Keep the exact class/method name and signature. Output "
        f"ONLY the corrected code in one fenced block."
    )


def swap_code_block(text: str, new_code: str, language_label: str) -> str:
    """Replace the LARGEST fenced code block in `text` with `new_code` (keeping a
    language fence). Used to drop the verified, corrected solution INTO the answer
    in place of the buggy one — so the user sees one clean, working program, not
    a broken version followed by a fix. Appends a block if the answer had none."""
    tag = _fence_tag(language_label)
    fenced = f"```{tag}\n{new_code.strip()}\n```"
    blocks = list(re.finditer(r"```[a-zA-Z0-9+#]*\n.*?```", text or "", re.DOTALL))
    if not blocks:
        return (text or "").rstrip() + "\n\n" + fenced
    b = max(blocks, key=lambda m: len(m.group(0)))
    return (text[:b.start()] + fenced + text[b.end():])


# ── Differential (reference) testing ────────────────────────────────────────
# A solution that passes the 1-3 VISIBLE examples can still be wrong on the
# hidden/edge cases (the classic "accepted the examples, WA on submit"). We ask
# the model to write a self-contained fuzz harness — the candidate + an
# independent BRUTE-FORCE reference + a seeded random-input generator — run it
# ONCE in the sandbox, and read a concrete counterexample if the two disagree.

_DIFF_SYS = (
    "You write a self-contained STRESS TEST that compares a candidate coding "
    "solution against an independent brute-force reference on random inputs. "
    "Output ONLY code in one fenced block — no prose."
)

_MISMATCH_RE = re.compile(
    r"^MISMATCH\s+input=(.*?)\s+cand=(.*?)\s+ref=(.*)$", re.MULTILINE)
_TIMING_RE = re.compile(
    r"^TIMING\s+small_ms=(\d+)\s+large_ms=(\d+)\s+factor=(\d+)", re.MULTILINE)

# The boundary taxonomy the fuzz harness must cover in ADDITION to random inputs
# — the cases hand-written examples and pure random sampling both routinely miss.
_EDGE_TAXONOMY = (
    "the SMALLEST input the constraints allow (empty / size-0 or size-1); a "
    "single element; ALL-IDENTICAL elements; an already-sorted and a reverse-"
    "sorted input; the MINIMUM and MAXIMUM allowed values (including negatives "
    "and zero where the constraints permit); and one input at the LARGEST size "
    "the constraints allow (capped so the brute force still finishes quickly)")


def _diff_prompt(problem: str, solution: str, lang_label: str, k: int,
                 include_timing: bool = False) -> str:
    timing = ""
    if include_timing:
        timing = (
            "7. AFTER the checks above, IF FEASIBLE for this problem: build one "
            "valid input near the LARGEST allowed size (capped so the CANDIDATE "
            "alone runs in under ~2 seconds) and one input exactly 8x smaller; "
            "run ONLY the candidate on each, measure wall-clock milliseconds for "
            "each, and print EXACTLY one line:\n"
            "   TIMING small_ms=<int> large_ms=<int> factor=8\n"
            "   If timing is not feasible, simply omit this line.\n")
    return (
        f"Language: {lang_label}\n\n"
        f"Problem (verbatim from the source):\n{problem[:2000]}\n\n"
        f"Candidate solution (it already passes the visible examples):\n"
        f"```\n{solution[:4000]}\n```\n\n"
        f"Write ONE complete, self-contained {lang_label} program that STRESS-"
        f"TESTS the candidate against a brute-force reference:\n"
        "1. Include the candidate solution EXACTLY as given (same names, "
        "signature) — do not modify it.\n"
        "2. Implement a SECOND, independent BRUTE-FORCE reference for the SAME "
        "problem — the simplest obviously-correct version, even if slow.\n"
        f"3. Build test inputs that OBEY the problem's constraints and MUST "
        f"include these boundary cases: {_EDGE_TAXONOMY}. THEN add {k} more "
        "SMALL random inputs (fixed RNG seed, tiny sizes so the brute force is "
        "fast).\n"
        "4. For each input, compute the candidate result and the reference "
        "result and compare them (treat the output as an unordered collection "
        "ONLY if the problem allows any order).\n"
        "5. On the FIRST input where they differ, print EXACTLY one line and "
        "stop immediately:\n"
        "   MISMATCH input=<the input, compact one-line> cand=<candidate "
        "result, compact> ref=<reference result, compact>\n"
        "6. If all agree, print EXACTLY: ALL_AGREE\n"
        f"{timing}"
        "Print NOTHING else (besides the optional TIMING line). The program "
        "MUST compile and run as-is. Output ONLY the program in one fenced "
        "code block."
    )


_BIGO_RE = re.compile(r"O\s*\(\s*([^)]{1,40})\)", re.IGNORECASE)


def _claimed_time_complexity(answer_text: str) -> str | None:
    """Extract + normalize the FIRST claimed time complexity from the answer
    (e.g. 'O(n log n)' → 'nlogn'). Best-effort; None when none is stated."""
    if not answer_text:
        return None
    m = _BIGO_RE.search(answer_text)
    if not m:
        return None
    body = re.sub(r"\s+", "", m.group(1).lower())
    if body in ("1", "logn", "lgn", "loglogn"):
        return "sublinear"
    if body in ("n", "nlogn", "nlgn", "m+n", "n+m", "v+e", "e+v"):
        return "linearish"
    return "superlinear"   # n^2, n*m (as a grid), 2^n, n!, … → not linear-ish


def _complexity_advisory(answer_text: str, timing: dict,
                         flag_ratio: float) -> str | None:
    """A SOFT advisory (never a failure) when the measured growth looks far
    worse than the claimed Big-O. Deliberately conservative: only fires on a
    linear-ish claim with a trustworthy (non-tiny) small-run time and an
    observed size-ratio well above the input-size factor."""
    claim = _claimed_time_complexity(answer_text)
    if claim != "linearish":
        return None
    try:
        small = int(timing.get("small_ms", 0))
        large = int(timing.get("large_ms", 0))
        factor = max(2, int(timing.get("factor", 8)))
    except Exception:  # noqa: BLE001
        return None
    if small < 20 or large <= 0:      # too little signal to trust the ratio
        return None
    ratio = large / small
    if ratio > factor * float(flag_ratio):
        return (f"\nℹ️ *Performance note:* on an input {factor}× larger the "
                f"solution took ~{ratio:.0f}× longer — that grows faster than the "
                f"stated linear-ish complexity, so the claimed Big-O may be "
                f"optimistic. Worth a look.")
    return None


async def differential_check(problem_text: str, solution: str,
                             language_label: str, on_stage=None,
                             min_difficulty: str | None = None,
                             want_timing: bool = False) -> dict | None:
    """Stress-test `solution` against an LLM-written brute-force reference across
    a boundary taxonomy + random inputs. Returns a report dict:
      {"counterexample": {input, cand, ref} | None, "timing": {..} | None}
    `counterexample` is set on a confirmed disagreement; `timing` (only when
    `want_timing`) carries the optional TIMING probe. Returns None only when the
    harness couldn't be built/run at all — fail-open, so a passing solution is
    NEVER turned into a false failure."""
    try:
        from app.core.config_loader import cfg
        import asyncio
        from app.sandbox.executor import run_code
        from app.sandbox import lang_registry as _lr

        rt = runtime_for(language_label)
        if rt is None or not runtime_available(language_label):
            return None
        sandbox_lang, _ = rt
        _ver = _lr.version_from_label(language_label)
        k = int(getattr(cfg.code_solver, "differential_cases", 50))

        await _emit(on_stage, "Stress-testing against a reference solution")
        _tiers = _harness_tiers(min_difficulty)
        program = ""
        for _attempt in range(3):
            msgs = [{"role": "system", "content": _DIFF_SYS},
                    {"role": "user", "content": _diff_prompt(
                        problem_text, solution, language_label, k,
                        include_timing=want_timing)}]
            try:
                txt = await _stream_complete(
                    msgs, _tiers[_attempt], session_key=f"diff:{_attempt}")
            except Exception:  # noqa: BLE001 — an LLM outage isn't a verdict
                txt = ""
            program = _extract_code_block(txt or "") or (txt or "")
            if program.strip():
                break
        if not program.strip():
            return None

        run = await asyncio.to_thread(
            run_code, program, sandbox_lang, version=_ver)
        out = (run.stdout or "")
        report: dict = {"counterexample": None, "timing": None}
        m = _MISMATCH_RE.search(out)
        if m:
            report["counterexample"] = {
                "input": m.group(1).strip()[:200],
                "cand": m.group(2).strip()[:200],
                "ref": m.group(3).strip()[:200]}
        if want_timing:
            t = _TIMING_RE.search(out)
            if t:
                report["timing"] = {"small_ms": int(t.group(1)),
                                    "large_ms": int(t.group(2)),
                                    "factor": int(t.group(3))}
        return report
    except Exception as exc:  # noqa: BLE001
        log.info("differential_check failed: %s", exc)
        return None


def _should_differential(examples: list[dict]) -> bool:
    """Run differential testing when enabled AND (forced OR the visible-example
    coverage is thin) — so well-covered problems stay fast."""
    try:
        from app.core.config_loader import cfg
        cs = cfg.code_solver
        if not bool(getattr(cs, "differential_testing", True)):
            return False
        if bool(getattr(cs, "differential_always", False)):
            return True
        return len(examples or []) <= int(
            getattr(cs, "differential_thin_examples", 2))
    except Exception:  # noqa: BLE001
        return False


async def _harden_against(problem_text: str, solution: str, language_label: str,
                          examples: list[dict], counterexample: dict,
                          min_difficulty: str | None, on_stage=None):
    """A differential counterexample was found on a solution that PASSED the
    visible examples. Try to repair it, then RE-VERIFY against the visible
    examples — only accept the fix if it still passes them all (so a possibly
    imperfect reference can never replace a visible-passing solution with worse
    code). Returns (verdict_suffix, corrected_code) on success, else (None, None).
    """
    try:
        await _emit(on_stage, "Hardening the solution against the edge case")
        ce = (f"case 1: input={counterexample.get('input')} "
              f"got={counterexample.get('cand')} "
              f"want={counterexample.get('ref')}")
        msgs = [
            {"role": "system", "content": _REPAIR_SYS},
            {"role": "user", "content": _repair_prompt(
                problem_text, solution, language_label,
                {"details": [ce],
                 "reason": "a stress test against a reference solution found "
                           "this counterexample"})},
        ]
        fix_text = await _stream_complete(
            msgs, (min_difficulty or "standard"), session_key="diff-harden")
        corrected = _extract_code_block(fix_text or "") or (fix_text or "")
        if not corrected.strip():
            return None, None
        v2 = await verify_solution(
            problem_text, f"```\n{corrected}\n```", language_label,
            examples, on_stage=on_stage, min_difficulty=min_difficulty)
        if v2.get("status") == "passed":
            gen = (" (tests derived from the problem)"
                   if v2.get("generated") else "")
            return (
                f"\n\n---\n✅ **Verified in sandbox** — passed all "
                f"{v2['total']} example{'s' if v2['total'] != 1 else ''} "
                f"({language_label}){gen}; also **hardened against an edge case** "
                f"a reference stress-test found.",
                corrected)
        return None, None
    except Exception as exc:  # noqa: BLE001
        log.info("_harden_against failed: %s", exc)
        return None, None


async def verify_and_maybe_repair(problem_text: str, answer_text: str,
                                  language_label: str,
                                  examples: list[dict] | None = None,
                                  max_repairs: int = 1,
                                  on_stage=None,
                                  min_difficulty: str | None = None,
                                  ) -> tuple[str, str | None]:
    """Verify the answer's solution; if it FAILS the examples (or won't compile),
    regenerate a fix and re-verify, up to `max_repairs` times.

    Returns `(verdict_markdown, corrected_code)`:
      • verdict_markdown — the ✅/⚠️/ℹ️ line to append to the answer;
      • corrected_code   — the FINAL, verified solution when a repair succeeded
        (so the caller can swap it into the answer in place of the buggy one),
        else None.
    Never raises.
    """
    try:
        examples = examples or extract_examples(problem_text)
        v = await verify_solution(problem_text, answer_text, language_label,
                                  examples, on_stage=on_stage,
                                  min_difficulty=min_difficulty)
        # Fast gate PASSED → optionally stress-test against a reference to catch
        # hidden/edge-case bugs the visible examples missed. A found counter-
        # example only HARDENS the solution (a repair that still passes every
        # visible example is swapped in); a correct solution is NEVER downgraded
        # to a false failure — if hardening can't beat the visible bar we keep the
        # original and report the honest ✅.
        _extra_note = ""
        if v.get("status") == "passed" and _should_differential(examples):
            try:
                from app.core.config_loader import cfg as _cfgC
                _want_timing = bool(getattr(
                    _cfgC.code_solver, "complexity_check", False))
                _sol = _extract_code_block(answer_text)
                if _sol.strip():
                    _rep = await differential_check(
                        problem_text, _sol, language_label, on_stage=on_stage,
                        min_difficulty=min_difficulty, want_timing=_want_timing)
                    _ce = _rep.get("counterexample") if _rep else None
                    if _ce:
                        _hsuffix, _hcode = await _harden_against(
                            problem_text, _sol, language_label, examples, _ce,
                            min_difficulty, on_stage)
                        if _hcode:
                            return _hsuffix, _hcode
                        # Couldn't harden past the visible bar → keep the original
                        # (still passes all visible examples). Fall through to the
                        # normal passed verdict below.
                    elif _want_timing and _rep and _rep.get("timing"):
                        # Advisory-only complexity note (never a failure).
                        _adv = _complexity_advisory(
                            answer_text, _rep["timing"],
                            float(getattr(_cfgC.code_solver,
                                          "complexity_flag_ratio", 3.0)))
                        if _adv:
                            _extra_note = _adv
            except Exception:  # noqa: BLE001 — differential must never break a pass
                pass
        # Only genuine failures / compile crashes are repairable. Passed,
        # skipped (no runtime / no examples), and "no code" just report.
        if v.get("status") not in ("failed", "error"):
            return verdict_markdown(v, language_label) + _extra_note, None
        if v.get("status") == "error" and (
                "not installed" in v.get("reason", "")
                or "no examples" in v.get("reason", "")
                or "no code" in v.get("reason", "")):
            return verdict_markdown(v, language_label), None

        current = _extract_code_block(answer_text)
        first_fail = v
        last_v = v
        for _ in range(max(1, max_repairs)):
            try:
                await _emit(on_stage, "Fixing the code and re-testing")
                msgs = [
                    {"role": "system", "content": _REPAIR_SYS},
                    {"role": "user", "content": _repair_prompt(
                        problem_text, current, language_label, last_v)},
                ]
                # Streaming route (cross-provider failover) — same reason as the
                # harness build: the one-shot complete_routed dies with
                # NoRouteAvailable when the free models are momentarily busy.
                fix_text = await _stream_complete(
                    msgs, (min_difficulty or "standard"),
                    session_key="verify-repair")
                corrected = _extract_code_block(fix_text or "") or (fix_text or "")
                if not corrected.strip():
                    break
                v2 = await verify_solution(
                    problem_text, f"```\n{corrected}\n```", language_label,
                    examples, on_stage=on_stage, min_difficulty=min_difficulty)
                if v2.get("status") == "passed":
                    # The caller swaps `corrected` into the answer; the verdict is
                    # just the clean ✅ line noting it was auto-corrected.
                    what = ("a wrong result"
                            if first_fail.get("status") == "failed"
                            else "an error")
                    gen = (" (tests derived from the problem)"
                           if v2.get("generated") else "")
                    # A repair that passes the VISIBLE examples can still miss a
                    # hidden case. Stress-test it once and, if a counterexample
                    # turns up, harden further (bounded, gated) — only accepting a
                    # fix that still passes every visible example.
                    if _should_differential(examples):
                        try:
                            _r2 = await differential_check(
                                problem_text, corrected, language_label,
                                on_stage=on_stage, min_difficulty=min_difficulty)
                            _c2 = _r2.get("counterexample") if _r2 else None
                            if _c2:
                                _hs, _hc = await _harden_against(
                                    problem_text, corrected, language_label,
                                    examples, _c2, min_difficulty, on_stage)
                                if _hc:
                                    return _hs, _hc
                        except Exception:  # noqa: BLE001
                            pass
                    return (
                        f"\n\n---\n✅ **Verified in sandbox** — passed all "
                        f"{v2['total']} example"
                        f"{'s' if v2['total'] != 1 else ''} ({language_label}){gen}"
                        f"; auto-corrected {what}.",
                        corrected)
                current, last_v = corrected, v2
            except Exception:  # noqa: BLE001
                break
        # Repair didn't reach green — report the honest best (no swap).
        return verdict_markdown(last_v, language_label), None
    except Exception as exc:  # noqa: BLE001
        log.info("verify_and_maybe_repair failed: %s", exc)
        return "", None


def verdict_markdown(v: dict, language_label: str) -> str:
    """A short, user-facing verdict line appended to the answer."""
    st = v.get("status")
    total = v.get("total", 0)
    passed = v.get("passed", 0)
    # When the problem had no worked examples, the cases were derived — say so.
    gen = " (tests derived from the problem)" if v.get("generated") else ""
    if st == "passed":
        return (f"\n\n---\n✅ **Verified in sandbox** — passed all {total} "
                f"example{'s' if total != 1 else ''} ({language_label}){gen}.")
    if st == "failed":
        det = "; ".join(v.get("details", [])[:3])
        return (f"\n\n---\n⚠️ **Sandbox check: {passed}/{total} examples passed"
                f"{gen}.** {det}\nThe solution above may be incorrect — "
                f"double-check it.")
    if st == "error":
        return (f"\n\n---\nℹ️ Could not sandbox-verify: {v.get('reason', '')}")
    # skipped
    reason = v.get("reason", "")
    return f"\n\n---\nℹ️ Not sandbox-verified — {reason}." if reason else ""
