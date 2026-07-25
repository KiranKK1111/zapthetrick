"""Stage-6 §4.4 — live coding two-lane gate: split, complexity, honest badges."""
from __future__ import annotations

from app.live import two_lane as T

_ANSWER = (
    "My approach: I use a hash map to get O(n) lookups.\n\n"
    "```python\n"
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, v in enumerate(nums):\n"
    "        if target - v in seen:\n"
    "            return [seen[target - v], i]\n"
    "        seen[v] = i\n"
    "```\n\n"
    "Edge cases: empty input and duplicate values."
)


class TestSplitLanes:
    def test_splits_prose_and_code(self):
        s = T.split_lanes(_ANSWER)
        assert s.has_code is True
        assert s.language == "python"
        assert "def two_sum" in s.code
        # Prose keeps the approach + edge cases, drops the fenced code.
        assert "My approach" in s.prose and "Edge cases" in s.prose
        assert "def two_sum" not in s.prose

    def test_no_code_is_all_prose(self):
        s = T.split_lanes("Just talk about the tradeoffs, no code here.")
        assert s.has_code is False and s.code == ""
        assert "tradeoffs" in s.prose

    def test_longest_block_is_the_code_lane(self):
        ans = "```python\nx=1\n```\nmid\n```python\ndef big():\n    return 42\n```"
        assert "def big" in T.split_lanes(ans).code

    def test_never_raises(self):
        assert T.split_lanes(None).has_code is False   # type: ignore[arg-type]


class TestComplexityNote:
    def test_nested_loops_are_quadratic(self):
        code = "for i in a:\n    for j in b:\n        print(i, j)"
        assert "O(n" in T.complexity_note(code)
        assert "n²" in T.complexity_note(code) or "n^2" in T.complexity_note(code) \
            or "n2" in T.complexity_note(code) or "²" in T.complexity_note(code)

    def test_single_loop_is_linear_ish(self):
        note = T.complexity_note("for i in a:\n    print(i)")
        assert "O(n" in note

    def test_no_loops_is_constant(self):
        note = T.complexity_note("x = 1\nreturn x + 2")
        assert "O(1)" in note

    def test_advisory_wording(self):
        assert "advisory" in T.complexity_note("for i in a:\n    pass").lower()

    def test_empty_is_blank(self):
        assert T.complexity_note("") == ""


class TestRevealBadge:
    def test_verified_says_compiled_and_ran(self):
        b = T.reveal_badge("verified", "python", examples=3)
        assert "Compiled & ran" in b and "python" in b and "3 example" in b

    def test_repaired_flag(self):
        assert "fixed a runtime error" in T.reveal_badge(
            "verified", "go", repaired=True)

    def test_not_run_is_honest_not_verified(self):
        b = T.reveal_badge("not_run", "rust")
        assert "Not executed" in b
        assert "Compiled" not in b and "verified" not in b.lower()

    def test_failed_is_not_executed(self):
        assert "Not executed" in T.reveal_badge("failed", "java")

    def test_unavailable_is_honest(self):
        assert "Not executed" in T.reveal_badge("unavailable", "python")


class TestCompose:
    def test_reassembles_prose_code_badge(self):
        out = T.compose("My approach.", "print(1)", "python", "OK badge")
        assert "My approach." in out
        assert "```python" in out and "print(1)" in out
        assert "_OK badge_" in out

    def test_prose_only_when_no_code(self):
        out = T.compose("Just prose.", "", "", "")
        assert out == "Just prose."


class TestFlag:
    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "two_lane", False, raising=False)
        assert T.enabled() is False

    def test_enabled_reads_flag(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "two_lane", True, raising=False)
        assert T.enabled() is True
