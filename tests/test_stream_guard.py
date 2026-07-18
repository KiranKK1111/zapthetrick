"""Mid-stream output guards (user report 2026-07-09: "unwanted, messy,
never-ending responses from some models") — the incremental reasoning
scrubber and the repetition/ceiling kill switch."""
from __future__ import annotations

import asyncio

from app.llm.stream_guard import RepetitionGuard, StreamScrubber


class TestStreamScrubber:
    def _run(self, chunks: list[str]) -> str:
        s = StreamScrubber()
        return "".join(s.feed(c) for c in chunks) + s.flush()

    def test_think_block_removed_across_chunks(self):
        out = self._run(["Hello <thi", "nk>secret", " chain</think> world"])
        assert out == "Hello  world"

    def test_thinking_and_reasoning_tags(self):
        assert self._run(["<thinking>x</thinking>ok"]) == "ok"
        assert self._run(["<reasoning>y</reasoning>ok"]) == "ok"

    def test_unterminated_think_dropped(self):
        assert self._run(["ok <think>never closed..."]) == "ok "

    def test_harmony_final_kept_analysis_dropped(self):
        out = self._run([
            "<|channel|>analysis<|message|>let me think<|end|>",
            "<|channel|>final<|message|>The answer.",
        ])
        assert out == "The answer."

    def test_bare_harmony_tokens_stripped(self):
        assert self._run(["A<|end|>B<|start|>C"]) == "ABC"

    def test_code_with_angle_brackets_survives(self):
        out = self._run(["List<int> xs; if (a < b) return;"])
        assert out == "List<int> xs; if (a < b) return;"

    def test_plain_text_unchanged(self):
        text = "Normal answer with **markdown** and `code`."
        assert self._run([text]) == text

    def test_stray_close_tag_dropped(self):
        assert self._run(["hello</think> world"]) == "hello world"


class TestRepetitionGuard:
    def test_consecutive_repeat_killed(self):
        g = RepetitionGuard(max_repeats=3)
        sent = "I will repeat this exact sentence forever and ever. "
        assert any(g.feed(sent) for _ in range(8))

    def test_varied_prose_passes(self):
        g = RepetitionGuard(max_repeats=3)
        for s in ("First idea here. ", "A different second point. ",
                  "Third distinct thought. ", "Fourth new angle. ") * 5:
            assert not g.feed(s)

    def test_char_ceiling(self):
        g = RepetitionGuard(max_repeats=3, max_chars=100)
        assert g.feed("x" * 200)

    def test_short_fragments_ignored(self):
        g = RepetitionGuard(max_repeats=3)
        for _ in range(10):
            assert not g.feed("Yes.\n")     # under min_unit_len


class TestGuardedStreamWiring:
    def test_llm_client_wraps_stream(self):
        from app.core.llm_client import LLMClient
        client = LLMClient()

        async def fake_gen():
            yield "Visible <think>hidden</think>text. "
            for _ in range(10):
                yield "This sentence repeats verbatim every single time. "

        async def collect():
            return [c async for c in client._guarded_stream(fake_gen())]

        chunks = asyncio.run(collect())
        joined = "".join(chunks)
        assert "hidden" not in joined
        assert "Visible" in joined
        assert "repeating itself" in joined     # kill-switch note appended
        # far fewer than 10 repeats made it through
        assert joined.count("repeats verbatim") <= 4
