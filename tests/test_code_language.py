"""Language detection from vision-read coding-problem screenshots."""
from app.codeintel.code_language import (
    detect_language,
    detect_language_label,
    looks_like_coding_problem,
    requested_language,
)


class TestRequestedLanguage:
    """The user's explicitly asked-for language OUTRANKS the image read."""

    def test_cue_phrases(self):
        assert requested_language("solve this in swift") == "swift"
        assert requested_language("give me the Dart solution") == "dart"
        assert requested_language("in swift please") == "swift"

    def test_bare_terse_request(self):
        # A terse composer entry alongside a screenshot ("swift", "use rust").
        assert requested_language("swift") == "swift"
        assert requested_language("use rust") == "rust"
        assert requested_language("dart") == "dart"

    def test_symbolic_and_versioned_names(self):
        assert requested_language("c#") == "csharp"
        assert requested_language("golang") == "go"
        assert requested_language("python2") == "python"

    def test_prose_not_a_request(self):
        assert requested_language("Solve this.") is None
        assert requested_language("go through this class and explain") is None
        # A long sentence that merely mentions a language is not a request via
        # the bare-name fallback (length-guarded).
        assert requested_language(
            "please read the whole file and summarize the rust module for me "
            "in a couple of short paragraphs") is None


class TestVersionLabel:
    def test_python3_vs_python2(self):
        assert detect_language_label("Python3 Auto class Solution:") == "Python3"
        assert detect_language_label("Python2 class Solution:") == "Python2"

    def test_bare_python_defaults_to_py3(self):
        assert detect_language_label("Python Auto class Solution:") == "Python3"

    def test_rust_label(self):
        assert detect_language_label("Rust Auto impl Solution") == "Rust"
        # from stub only (no word "Rust")
        assert detect_language_label("impl Solution { pub fn trap") == "Rust"

    def test_cpp_and_csharp_labels(self):
        assert detect_language_label("C++ vector<int>") == "C++"
        assert detect_language_label("C# Console.WriteLine") == "C#"

    def test_none_when_unknown(self):
        assert detect_language_label("Invoice total 100") is None


class TestExplicitLanguage:
    def test_leetcode_java_chip_and_stub(self):
        # The exact shape the local vision model reads off a LeetCode screenshot.
        t = ("Java Auto class Solution|public int "
             "firstMissingPositive(int[] nums) { return 0; }")
        assert detect_language(t) == "java"

    def test_python3_selector(self):
        assert detect_language("Python3  class Solution:") == "python"

    def test_cpp_plus_plus(self):
        assert detect_language("Language: C++  vector<int> nums") == "cpp"

    def test_typescript_before_javascript(self):
        assert detect_language("TypeScript  const f = () => 1") == "typescript"


class TestStubInference:
    def test_python_stub(self):
        assert detect_language("class Solution:\n    def f(self, nums): pass") == "python"

    def test_java_stub(self):
        assert detect_language("public int firstMissingPositive(int[] nums) {") == "java"

    def test_cpp_stub(self):
        assert detect_language("#include <vector>\nusing std::vector;") == "cpp"

    def test_go_stub(self):
        assert detect_language("func firstMissing(nums []int) int { return 0 }") == "go"

    def test_scala_stub_not_python(self):
        # Regression: Scala's `def` used to match the Python stub. `object` +
        # typed def + Array[] must resolve to scala even without the chip.
        code = "object Solution {\n  def trap(height: Array[Int]): Int = {"
        assert detect_language(code) == "scala"

    def test_scala_chip(self):
        assert detect_language("Scala Auto object Solution") == "scala"
        assert detect_language_label("Scala") == "Scala"

    def test_racket_and_clojure(self):
        assert detect_language("Racket") == "racket"
        assert detect_language("Clojure") == "clojure"

    def test_python_still_detects(self):
        assert detect_language("class Solution:\n    def f(self): pass") == "python"

    def test_no_language(self):
        assert detect_language("Invoice #4471 Total 1240.00") is None
        assert detect_language("") is None


class TestChipAutoDetection:
    """The '<Lang> Auto' editor chip — reliable for short/symbol names."""

    def test_symbol_chips(self):
        # These failed before: `+`/`#` break the trailing \b, and C/R/D/V are
        # single letters skipped in prose — the "Auto" context makes them safe.
        assert detect_language("C Auto int main()") == "c"
        assert detect_language("C++ Auto stuff") == "cpp"
        assert detect_language("C# Auto stuff") == "csharp"
        assert detect_language("R Auto stuff") == "r"
        assert detect_language("F# Auto stuff") == "fsharp"
        assert detect_language("D Auto stuff") == "d"
        assert detect_language("V Auto stuff") == "vlang"

    def test_multiword_chips(self):
        assert detect_language("Objective-C Auto") == "objc"
        assert detect_language("Common Lisp Auto") == "lisp"
        assert detect_language("Visual Basic Auto") == "vbnet"

    def test_full_registry_chip_audit(self):
        # Every registry language must be detected from its display-label chip.
        from app.codeintel.code_language import _CANON_TO_LABEL
        from app.sandbox import lang_registry as lr
        misses = []
        for cid in lr.supported_ids():
            chip = _CANON_TO_LABEL.get(cid, cid)
            if detect_language(f"{chip} Auto some problem text") != cid:
                misses.append(cid)
        assert not misses, f"chip detection missed: {misses}"

    def test_no_false_positive_on_prose(self):
        assert detect_language(
            "Given an array, return the sum. Example: Input [1,2] Output 3") is None


class TestStubDisambiguation:
    def test_c_vs_cpp(self):
        assert detect_language("#include <stdio.h>\nint f(){ printf(); }") == "c"
        assert detect_language("#include <vector>\nusing std::vector; cout") == "cpp"

    def test_csharp_vs_java(self):
        assert detect_language("public class S { void f(){ Console.WriteLine(); }}") == "csharp"
        assert detect_language("public class S { void f(){ System.out.println(); }}") == "java"

    def test_ruby_vs_python(self):
        assert detect_language("def trap(height)\n  puts height\nend") == "ruby"
        assert detect_language("def trap(self, height):\n    return 0") == "python"

    def test_chip_on_separate_line_from_auto(self):
        # OCR often puts the chip and "Auto" on separate lines; the 2-word
        # capture must not swallow "Auto".
        assert detect_language("Erlang\nAuto\n-spec f() ->") == "erlang"
        assert detect_language("Rust\nAuto\nimpl Solution") == "rust"

    def test_functional_language_stubs(self):
        # LeetCode functional langs — detect from code when the chip is missed.
        assert detect_language(
            "-spec multiply(N :: unicode:unicode_binary()) ->\n"
            "  unicode:unicode_binary().\nmultiply(A, B) -> ok.") == "erlang"
        assert detect_language("defmodule Solution do\n  def multiply(a, b) do") == "elixir"
        assert detect_language("#lang racket\n(define (multiply a b)") == "racket"
        assert detect_language("(defn multiply [a b]") == "clojure"


class TestCodingProblem:
    def test_leetcode_problem(self):
        t = ("41. First Missing Positive. Given an unsorted integer array nums. "
             "Example 1: Input: nums = [1,2,0] Output: 3 Constraints:")
        assert looks_like_coding_problem(t) is True

    def test_class_solution_marker(self):
        assert looks_like_coding_problem("class Solution { }") is True

    def test_plain_text_not_a_problem(self):
        assert looks_like_coding_problem("Invoice #4471 Total 1240.00") is False
        assert looks_like_coding_problem("") is False


class TestOcrVsHallucination:
    """The core of the capability-aware fix: a Dart screenshot must resolve to
    dart even though the vision model hallucinated a C# solution. Language MUST
    be detected on the OCR text and the VLM text SEPARATELY (never merged) — the
    OCR read is exact, the VLM read hallucinates the language."""

    OCR_DART = ("Permutations II\nDart\nclass Solution {\n"
                "  List<List<int>> permuteUnique(List<int> nums) {\n  }\n}")
    VLM_CSHARP = ("The user selected a coding problem. Solution:\n"
                  "using System;\nusing System.Collections.Generic;\n"
                  "public class Solution {\n"
                  "  public IList<IList<int>> PermuteUnique(int[] nums) {\n"
                  "    Console.WriteLine(); } }")

    def test_ocr_reads_dart(self):
        assert detect_language(self.OCR_DART) == "dart"

    def test_vlm_hallucination_reads_csharp(self):
        # This is exactly why merging OCR+VLM and detecting on the blob broke:
        assert detect_language(self.VLM_CSHARP) == "csharp"

    def test_capable_cross_check_prefers_ocr(self):
        # capability-aware resolution (mirrors routes_attachments): trust the
        # capable model, but OCR's definitive read overrides a disagreement.
        ocr_lang = detect_language(self.OCR_DART)
        vlm_lang = detect_language(self.VLM_CSHARP)
        img_lang = vlm_lang or ocr_lang
        if ocr_lang and vlm_lang and ocr_lang != vlm_lang:
            img_lang = ocr_lang
        assert img_lang == "dart"

    def test_small_model_ocr_authoritative(self):
        ocr_lang = detect_language(self.OCR_DART)
        vlm_lang = detect_language(self.VLM_CSHARP)
        assert (ocr_lang or vlm_lang) == "dart"

    # OCR that read the Dart CODE but MISSED the tiny "Dart" chip word — still
    # resolves to dart via the `List<int>` stub (so the fix is robust even when
    # the chip glyph is dropped, as long as the code is read).
    OCR_DART_CODEONLY = ("class Solution {\n"
                         "  List<List<int>> permuteUnique(List<int> nums) {\n  }\n}")

    def test_ocr_codeonly_still_dart(self):
        assert detect_language(self.OCR_DART_CODEONLY) == "dart"

    def test_merged_blob_would_regress(self):
        # The bug: with the chip word missed, detecting on the MERGED (OCR code +
        # hallucinated C#) text lets csharp win (its stub is checked before
        # dart's). This is exactly why we detect on OCR and VLM SEPARATELY.
        merged = self.OCR_DART_CODEONLY + "\n\n" + self.VLM_CSHARP
        assert detect_language(merged) == "csharp"  # the thing we must NOT do
