"""Sandbox verification of image coding solutions — non-LLM parts."""
import asyncio

from app.codeintel import solution_verify as sv


class TestExtractExamples:
    def test_input_output_pairs(self):
        text = ("Example 1:\nInput: height = [4,2,0,3,2,5]\nOutput: 9\n"
                "Example 2:\nInput: nums = [1,2]\nOutput: 3\nConstraints:")
        ex = sv.extract_examples(text)
        assert len(ex) == 2
        assert ex[0]["input"] == "height = [4,2,0,3,2,5]"
        assert ex[0]["expected"] == "9"
        assert ex[1]["expected"] == "3"

    def test_stops_at_explanation(self):
        text = ("Input: nums = [1,2,0]\nOutput: 3\n"
                "Explanation: the numbers in range are present.")
        ex = sv.extract_examples(text)
        assert ex[0]["expected"] == "3"  # not the explanation

    def test_none_when_no_examples(self):
        assert sv.extract_examples("just some prose") == []
        assert sv.extract_examples("") == []

    def test_skips_input_output_format_section_headers(self):
        # The HackerRank currency-formatter bug: "Input Format" / "Output Format"
        # section headers were parsed as an example, grabbing the prose "On the
        # first line, print US: u where…" as the expected output → every correct
        # solution failed 0/N. Must bind to Sample Input/Output instead.
        text = ("Input Format\nA single double-precision number denoting payment.\n"
                "Constraints\n0 <= payment <= 10^9\n"
                "Output Format\nOn the first line, print US: u where u is payment "
                "formatted for US currency.\n"
                "Sample Input\n12324.134\n"
                "Sample Output\nUS: $12,324.13\nIndia: Rs.12,324.13\n"
                "Explanation\nEach line contains the value.")
        ex = sv.extract_examples(text)
        assert len(ex) == 1
        assert ex[0]["input"] == "12324.134"
        assert ex[0]["expected"].startswith("US: $12,324.13")
        assert "Format" not in ex[0]["expected"]
        assert "where" not in ex[0]["expected"]


class TestRuntimeDetection:
    def test_python_variants_map_to_python(self):
        assert sv.runtime_for("Python3")[0] == "python"
        assert sv.runtime_for("Python2")[0] == "python"
        assert sv.runtime_for("python")[0] == "python"

    def test_language_map(self):
        assert sv.runtime_for("Rust")[0] == "rust"
        assert sv.runtime_for("C++")[0] == "cpp"
        assert sv.runtime_for("Java")[0] == "java"
        assert sv.runtime_for("JavaScript")[0] == "javascript"

    def test_unknown_language(self):
        assert sv.runtime_for("Klingon") is None


class TestVerifyStatusLogic:
    """verify_solution's pass/fail decision (real body, mocked LLM + sandbox)."""

    def _verify(self, monkeypatch, stdout, run_status="ok", examples=2):
        import app.core.llm_client as lc
        import app.sandbox.executor as ex

        import types

        # The harness build now goes through the STREAMING path (stream_chat),
        # so mock that (an async generator) rather than complete_routed.
        async def fake_stream(msgs, model=None, session_key=None, options=None):
            yield "```python\nprint('harness')\n```"
        monkeypatch.setattr(lc.llm, "stream_chat", fake_stream)
        monkeypatch.setattr(sv, "runtime_available", lambda lang: True)

        r = types.SimpleNamespace(
            status=run_status, stdout=stdout, stderr="", reason="")
        monkeypatch.setattr(ex, "run_code", lambda *a, **k: r)
        exs = [{"input": str(i), "expected": str(i)} for i in range(examples)]
        return asyncio.run(sv.verify_solution(
            "p", "```python\nsol\n```", "Python3", examples=exs))

    def test_all_cases_pass_even_when_harness_runs_extra(self, monkeypatch):
        # Harness ran 3 PASS cases though only 2 examples were extracted — an
        # all-green run must be "passed", not a false "failed" (the count-mismatch
        # bug that made a correct Erlang solution report failed).
        v = self._verify(monkeypatch,
                         "CASE 1 PASS\nCASE 2 PASS\nCASE 3 PASS\n", examples=2)
        assert v["status"] == "passed"
        assert v["passed"] == 3 and v["total"] == 3

    def test_a_real_failure_is_reported(self, monkeypatch):
        v = self._verify(monkeypatch,
                         "CASE 1 PASS\nCASE 2 FAIL got=5 want=6\n", examples=2)
        assert v["status"] == "failed"
        assert v["passed"] == 1 and v["total"] == 2


class TestDeterministicPythonHarness:
    """The common `class Solution` Python shape builds a harness with NO LLM."""

    def test_builds_common_shape(self):
        sol = ("class Solution:\n    def twoSum(self, nums, target):\n"
               "        return [0, 1]")
        h = sv._python_harness(
            sol, [{"input": "nums = [2,7], target = 9", "expected": "[0,1]"}])
        assert h and "class Solution" in h and "CASE" in h

    def test_inplace_mutation_shape(self):
        rot = ("class Solution:\n    def rotate(self, matrix):\n"
               "        matrix.reverse()")
        h = sv._python_harness(
            rot, [{"input": "matrix = [[1,2],[3,4]]", "expected": "[[3,1],[4,2]]"}])
        assert h and "_got is None" in h        # in-place fallback present

    def test_fallback_custom_types(self):
        assert sv._python_harness(
            "class Solution:\n    def f(self, root):\n        return root  # TreeNode",
            [{"input": "root = []", "expected": "[]"}]) is None

    def test_fallback_multi_method(self):
        sol = ("class Solution:\n    def a(self, x): return x\n"
               "    def b(self, x): return x")
        assert sv._python_harness(sol, [{"input": "x = 1", "expected": "1"}]) is None

    def test_fallback_unparseable_input(self):
        assert sv._python_harness(
            "class Solution:\n    def f(self, n): return n",
            [{"input": "n = some_variable", "expected": "1"}]) is None

    def test_split_top_commas(self):
        assert sv._split_top_commas("nums = [1,2,3], target = 9") == \
            ["nums = [1,2,3]", "target = 9"]

    def test_parse_input_kwargs_and_positional(self):
        assert sv._parse_example_input("nums = [1,2], target = 3") == \
            ("kw", {"nums": [1, 2], "target": 3})
        assert sv._parse_example_input("[1,2,3]") == ("pos", [[1, 2, 3]])
        assert sv._parse_example_input("n = some_var") is None


class TestPythonFastPathNoLLM:
    """A Python function-style problem uses the deterministic harness — the LLM
    harness (_stream_complete) is never called."""

    def test_deterministic_path_skips_llm(self, monkeypatch):
        import types
        import app.sandbox.executor as ex_mod
        monkeypatch.setattr(sv, "runtime_available", lambda lang: True)

        async def boom(*a, **k):
            raise AssertionError("LLM harness must not run on the python fast path")
        monkeypatch.setattr(sv, "_stream_complete", boom)
        monkeypatch.setattr(ex_mod, "run_code", lambda *a, **k: types.SimpleNamespace(
            status="ok", stdout="CASE 1 got=[0,1] want=[0,1]\n", stderr="",
            reason=""))
        sol = ("```python\nclass Solution:\n    def twoSum(self, nums, target):\n"
               "        return [0, 1]\n```")
        v = asyncio.run(sv.verify_solution(
            "p", sol, "Python3",
            examples=[{"input": "nums = [2,7], target = 9", "expected": "[0,1]"}]))
        assert v["status"] == "passed" and v["passed"] == 1


class TestStdinBatchVerify:
    """stdin-style problems compile ONCE and run the artifact against every
    example via run_batch (not a full compile+run per example)."""

    def _run(self, monkeypatch, outputs, run_status="ok"):
        import app.sandbox.executor as ex_mod
        import types
        monkeypatch.setattr(sv, "runtime_available", lambda lang: True)
        cap = {"calls": 0, "stdins": None}

        def fake_batch(code, language, *, files=None, limits=None,
                       version=None, stdins=None):
            cap["calls"] += 1
            cap["stdins"] = list(stdins or [])
            return [types.SimpleNamespace(status=run_status, stdout=o,
                                          stderr="", reason="") for o in outputs]

        monkeypatch.setattr(ex_mod, "run_batch", fake_batch)
        sol = "```python\nx = input()\nprint(int(x) * 2)\n```"   # reads stdin
        exs = [{"input": "3", "input_raw": "3", "expected": "6"},
               {"input": "5", "input_raw": "5", "expected": "10"}]
        v = asyncio.run(sv.verify_solution("p", sol, "Python3", examples=exs))
        return v, cap

    def test_all_pass_one_compile(self, monkeypatch):
        v, cap = self._run(monkeypatch, ["6", "10"])
        assert v["status"] == "passed" and v["passed"] == 2 and v["total"] == 2
        assert cap["calls"] == 1                   # ONE batch (compiled once)
        assert cap["stdins"] == ["3\n", "5\n"]     # each example's stdin fed

    def test_a_failure_reported(self, monkeypatch):
        v, _ = self._run(monkeypatch, ["6", "999"])
        assert v["status"] == "failed" and v["passed"] == 1 and v["total"] == 2


class TestCodeExtraction:
    def test_last_longest_block(self):
        text = ("intro ```py\nx=1\n``` then ```python\n"
                "class Solution:\n    def f(self): return 1\n```")
        code = sv._extract_code_block(text)
        assert "class Solution" in code

    def test_no_block(self):
        assert sv._extract_code_block("no code here") == ""


class TestVerdictMarkdown:
    def test_passed(self):
        md = sv.verdict_markdown(
            {"status": "passed", "passed": 3, "total": 3}, "Java")
        assert "Verified in sandbox" in md and "3" in md

    def test_failed(self):
        md = sv.verdict_markdown(
            {"status": "failed", "passed": 1, "total": 3,
             "details": ["case 2: got=5 want=6"]}, "Python3")
        assert "1/3" in md and "case 2" in md

    def test_skipped_reason(self):
        md = sv.verdict_markdown(
            {"status": "skipped", "reason": "Rust runtime not installed"},
            "Rust")
        assert "Not sandbox-verified" in md and "Rust runtime" in md

    def test_case_regex(self):
        out = "CASE 1 PASS\r\nCASE 2 FAIL got=5 want=6\r\n"
        cases = sv._CASE_RE.findall(out)
        assert len(cases) == 2
        assert cases[0][1] == "PASS"
        assert cases[1][1] == "FAIL"


class TestVerdictNoFalsePositive:
    """The runner-format harness prints got=/want=; PYTHON judges. A broken
    solution that returns [] must FAIL, not vacuously pass (the Elixir bug)."""

    def test_empty_result_fails_against_nonempty(self):
        assert sv._verdict("[]", "[[1,1,2],[1,2,1],[2,1,1]]") is False

    def test_any_order_collection_passes(self):
        assert sv._verdict("[[2,1,1],[1,1,2],[1,2,1]]",
                           "[[1,1,2],[1,2,1],[2,1,1]]") is True

    def test_partial_result_fails(self):
        assert sv._verdict("[[1,1,2],[1,2,1]]",
                           "[[1,1,2],[1,2,1],[2,1,1]]") is False

    def test_scalar_and_flat(self):
        assert sv._verdict("6", "6") is True
        assert sv._verdict("5", "6") is False
        assert sv._verdict("[1, 2, 3]", "[1,2,3]") is True

    def test_runner_parse_python_judges(self):
        # The harness printed a broken (empty) result → the PARSER (not the
        # harness) must mark it FAIL.
        out = ("CASE 1 got=[] want=[[1,1,2],[1,2,1],[2,1,1]]\n"
               "CASE 2 got=[[1,2],[2,1]] want=[[2,1],[1,2]]\n")
        cases = sv._cases_from_run(out)
        assert [c[1] for c in cases] == ["FAIL", "PASS"]

    def test_runner_parse_falls_back_to_self_judged(self):
        # A harness that ignored the runner format and self-judged still parses.
        out = "CASE 1 PASS\nCASE 2 FAIL got=5 want=6\n"
        cases = sv._cases_from_run(out)
        assert [c[1] for c in cases] == ["PASS", "FAIL"]


class TestHarnessTierSeeding:
    """The harness build seeds its model tier at the difficulty the ANSWER used,
    so a solution generated at hard/expert doesn't fail verification on rate-
    limited free 'standard' models."""

    def test_default_starts_standard(self):
        assert sv._harness_tiers(None) == ("standard", "hard", "expert")
        assert sv._harness_tiers("") == ("standard", "hard", "expert")

    def test_seed_hard_skips_weak_tier(self):
        assert sv._harness_tiers("hard") == ("hard", "expert", "expert")

    def test_seed_expert_stays_expert(self):
        assert sv._harness_tiers("expert") == ("expert", "expert", "expert")

    def test_trivial_and_unknown_floor_at_standard(self):
        assert sv._harness_tiers("trivial") == ("standard", "hard", "expert")
        assert sv._harness_tiers("banana") == ("standard", "hard", "expert")


class TestEdgeCaseHelpers:
    def test_reads_stdin_detects_input_styles(self):
        assert sv._reads_stdin("a = input().split()")
        assert sv._reads_stdin("Scanner sc = new Scanner(System.in);")
        assert sv._reads_stdin("std::cin >> n;")
        assert sv._reads_stdin("reader := bufio.NewReader(os.Stdin)")
        assert not sv._reads_stdin("def multiply(a, b): return a*b")
        assert not sv._reads_stdin("class Solution { int f(int[] nums){} }")

    def test_tolerant_output_match(self):
        assert sv._outputs_match('"6"\n', "6")           # quotes + newline
        assert sv._outputs_match("[1, 2]", "[1,2]")      # list spacing
        assert sv._outputs_match("1 2 3", "1\n2\n3")     # whitespace kind
        assert sv._outputs_match("  yes ", "yes")
        assert not sv._outputs_match("5", "6")           # a real mismatch stays

    def test_raw_input_preserved(self):
        ex = sv.extract_examples("Input: 6\n7\nOutput: 42\nConstraints:")
        assert ex and "\n" in ex[0]["input_raw"]         # newlines kept for stdin
        assert ex[0]["expected"] == "42"


class TestMultiFileCompile:
    def test_c_extra_sources_added(self):
        from app.sandbox import lang_registry as lr
        cmds = [["gcc", "-O2", "main.c", "-o", "prog"], ["./prog"]]
        out = lr.augment_multifile("c", "main.c", cmds, {"helper.c": "x", "note.txt": "y"})
        assert out[0] == ["gcc", "-O2", "main.c", "helper.c", "-o", "prog"]
        assert out[1] == ["./prog"]                       # run step untouched

    def test_no_extra_sources_noop(self):
        from app.sandbox import lang_registry as lr
        cmds = [["gcc", "main.c", "-o", "prog"], ["./prog"]]
        assert lr.augment_multifile("c", "main.c", cmds, {"data.txt": "x"}) == cmds
        # interpreted language → never touched
        assert lr.augment_multifile("python", "main.py",
                                    [["python3", "main.py"]], {"h.py": "x"}) \
            == [["python3", "main.py"]]


class TestSwapCodeBlock:
    def test_replaces_largest_block(self):
        text = ("Here's the solution:\n```python\ndef f(): return 5  # buggy\n```\n"
                "Explanation follows.")
        out = sv.swap_code_block(text, "def f(): return 6  # fixed", "Python3")
        assert "return 6" in out and "return 5" not in out
        assert "Explanation follows." in out          # prose preserved
        assert out.count("```") == 2                   # still one block

    def test_appends_when_no_block(self):
        out = sv.swap_code_block("just prose", "print(1)", "Python3")
        assert "```python" in out and "print(1)" in out


class TestDifferentialCheck:
    """The fuzz harness's MISMATCH/ALL_AGREE output is parsed into a
    counterexample; anything else is fail-open (None → never a false failure)."""

    def _check(self, monkeypatch, harness_out):
        import types
        import app.sandbox.executor as ex_mod
        monkeypatch.setattr(sv, "runtime_available", lambda lang: True)

        async def fake_stream(msgs, difficulty, session_key=None):
            return "```python\nprint('harness')\n```"
        monkeypatch.setattr(sv, "_stream_complete", fake_stream)
        monkeypatch.setattr(ex_mod, "run_code", lambda *a, **k: types.SimpleNamespace(
            stdout=harness_out, stderr="", status="ok"))
        return asyncio.run(sv.differential_check("prob", "sol", "Python3"))

    def test_mismatch_returns_counterexample(self, monkeypatch):
        rep = self._check(
            monkeypatch, "MISMATCH input=[3,1,2] cand=[] ref=[[1,2,3]]\n")
        assert rep["counterexample"] == {
            "input": "[3,1,2]", "cand": "[]", "ref": "[[1,2,3]]"}

    def test_all_agree_no_counterexample(self, monkeypatch):
        rep = self._check(monkeypatch, "ALL_AGREE\n")
        assert rep["counterexample"] is None

    def test_uncompilable_harness_is_failopen(self, monkeypatch):
        rep = self._check(monkeypatch, "error: could not compile")
        assert rep["counterexample"] is None


class TestComplexityAdvisory:
    """Opt-in, advisory-only: a super-linear measured growth on a linear-ish
    claim yields a soft note; anything ambiguous stays silent."""

    def test_flags_superlinear_growth_on_linear_claim(self):
        adv = sv._complexity_advisory(
            "Time Complexity: O(n) linear.",
            {"small_ms": 40, "large_ms": 40 * 70, "factor": 8}, 3.0)
        assert adv is not None and "faster than the stated" in adv

    def test_silent_when_growth_matches_linear_claim(self):
        adv = sv._complexity_advisory(
            "Time Complexity: O(n).",
            {"small_ms": 40, "large_ms": 40 * 8, "factor": 8}, 3.0)
        assert adv is None

    def test_silent_on_superlinear_claim(self):
        # Claimed O(n^2) → a quadratic measurement is expected, no advisory.
        adv = sv._complexity_advisory(
            "Time Complexity: O(n^2).",
            {"small_ms": 40, "large_ms": 40 * 70, "factor": 8}, 3.0)
        assert adv is None

    def test_silent_when_timing_too_noisy(self):
        adv = sv._complexity_advisory(
            "O(n)", {"small_ms": 3, "large_ms": 900, "factor": 8}, 3.0)
        assert adv is None   # small_ms < 20ms → untrustworthy, stay silent

    def test_complexity_extraction(self):
        assert sv._claimed_time_complexity("It's O(n log n) overall.") == "linearish"
        assert sv._claimed_time_complexity("O(1) constant") == "sublinear"
        assert sv._claimed_time_complexity("O(n^2)") == "superlinear"
        assert sv._claimed_time_complexity("no complexity here") is None


class TestShouldDifferential:
    def test_thin_coverage_runs(self):
        assert sv._should_differential([{}]) is True
        assert sv._should_differential([{}, {}]) is True

    def test_well_covered_skips(self):
        assert sv._should_differential([{}, {}, {}]) is False


class TestDifferentialHarden:
    """A passing solution + a differential counterexample: hardened ONLY if the
    fix still passes every visible example; otherwise the original passing
    solution and its honest ✅ are kept (never a false downgrade)."""

    def _run(self, monkeypatch, repaired_passes):
        calls = {"n": 0}

        async def fake_verify(prob, ans, lbl, ex=None, on_stage=None,
                              min_difficulty=None):
            calls["n"] += 1
            if calls["n"] == 1:   # fast gate
                return {"status": "passed", "passed": 1, "total": 1,
                        "details": [], "reason": ""}
            return {"status": "passed" if repaired_passes else "failed",
                    "passed": 1 if repaired_passes else 0, "total": 1,
                    "details": [], "reason": ""}
        monkeypatch.setattr(sv, "verify_solution", fake_verify)

        async def fake_diff(prob, sol, lbl, on_stage=None, min_difficulty=None,
                            want_timing=False):
            return {"counterexample": {"input": "a", "cand": "1", "ref": "2"},
                    "timing": None}
        monkeypatch.setattr(sv, "differential_check", fake_diff)

        async def fake_stream(msgs, difficulty, session_key=None):
            return "```python\nclass Solution:\n    pass\n```"
        monkeypatch.setattr(sv, "_stream_complete", fake_stream)

        return asyncio.run(sv.verify_and_maybe_repair(
            "prob\nInput: x\nOutput: 6", "```python\nclass Solution: pass\n```",
            "Python3", examples=[{"input": "x", "expected": "6"}]))

    def test_hardens_when_fix_passes_visible(self, monkeypatch):
        suffix, code = self._run(monkeypatch, repaired_passes=True)
        assert code is not None
        assert "hardened against an edge case" in suffix

    def test_keeps_original_when_fix_cannot_beat_visible(self, monkeypatch):
        suffix, code = self._run(monkeypatch, repaired_passes=False)
        assert code is None                       # original solution kept
        assert "Verified in sandbox" in suffix    # honest ✅, not a false ⚠️


class TestSelfRepair:
    """Orchestration of verify → repair → re-verify (LLM + sandbox mocked)."""

    def _run(self, monkeypatch, first_status, repair_result):
        verify_calls = {"n": 0}

        async def fake_verify(prob, ans, lbl, ex=None, on_stage=None,
                              min_difficulty=None):
            verify_calls["n"] += 1
            if verify_calls["n"] == 1:
                return {"status": first_status, "passed": 1, "total": 2,
                        "details": ["case 2: got=5 want=6"], "reason": "boom"}
            return repair_result

        async def fake_stream(msgs, model=None, session_key=None, options=None):
            yield "```python\nclass Solution:\n    def f(self): return 6\n```"

        monkeypatch.setattr(sv, "verify_solution", fake_verify)
        import app.core.llm_client as lc
        monkeypatch.setattr(lc.llm, "stream_chat", fake_stream)
        return asyncio.run(sv.verify_and_maybe_repair(
            "prob\nInput: x\nOutput: 6", "```python\nbad\n```", "Python3",
            examples=[{"input": "x", "expected": "6"}]))

    def test_repair_succeeds_returns_verified_fix(self, monkeypatch):
        # Repair succeeded → (clean ✅ verdict, the corrected code to swap in).
        suffix, fixed = self._run(monkeypatch, "failed",
                                  {"status": "passed", "passed": 2, "total": 2,
                                   "details": [], "reason": ""})
        assert "Verified in sandbox" in suffix
        assert "auto-corrected" in suffix
        assert fixed is not None and "return 6" in fixed  # the working code

    def test_passed_first_time_no_repair(self, monkeypatch):
        async def fake_verify(prob, ans, lbl, ex=None, on_stage=None,
                              min_difficulty=None):
            return {"status": "passed", "passed": 2, "total": 2,
                    "details": [], "reason": ""}
        monkeypatch.setattr(sv, "verify_solution", fake_verify)
        suffix, fixed = asyncio.run(sv.verify_and_maybe_repair(
            "p", "```python\nok\n```", "Python3",
            examples=[{"input": "x", "expected": "6"}]))
        assert "Verified in sandbox" in suffix and fixed is None

    def test_repair_fails_reports_honestly(self, monkeypatch):
        suffix, fixed = self._run(monkeypatch, "failed",
                                  {"status": "failed", "passed": 1, "total": 2,
                                   "details": ["case 2: got=5 want=6"],
                                   "reason": ""})
        assert "1/2" in suffix  # honest failure verdict
        assert fixed is None    # nothing to swap
