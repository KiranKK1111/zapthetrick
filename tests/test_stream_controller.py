"""Real-time (mid-stream) quality controller — `app/quality/stream_controller`
and its wiring into the one streaming choke point, `LLMClient._guarded_stream`.

The controller is the layer that catches what the stream guard cannot see:
refusal leakage, error/apology spikes, emptiness, and unpunctuated token-level
degeneration. It is sampled on a cadence and is deliberately reluctant to stop a
stream — a false-positive kill truncates a good answer.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core.config_loader import cfg
from app.core.llm_client import LLMClient, last_stream_verdict
from app.quality.stream_controller import (
    CONTINUE,
    FLAG,
    REGENERATE,
    QualityVerdict,
    assess_partial,
)

HEALTHY = (
    "Kafka keeps messages in an append-only commit log that is partitioned "
    "across brokers. Each partition has one leader and several followers, so "
    "producers write to the leader while consumers read at their own offset. "
    "Retention is time or size based rather than delivery based, which means a "
    "consumer group can replay history after a failure without asking anyone "
    "to resend anything. That decoupling is the whole point of the design."
)


def _drain(agen) -> str:
    async def go():
        return "".join([c async for c in agen])
    return asyncio.run(go())


def _drain_with_verdict(agen) -> tuple[str, object]:
    """Read the verdict INSIDE the consuming task — an async generator mutates
    the context of whoever iterates it (a route handler), which is exactly the
    caller that wants the flag."""
    async def go():
        text = "".join([c async for c in agen])
        return text, last_stream_verdict()
    return asyncio.run(go())


async def _gen(chunks):
    for c in chunks:
        yield c


class TestAssessPartial:
    def test_actions_are_the_documented_strings(self):
        assert (CONTINUE, FLAG, REGENERATE) == ("continue", "flag", "regenerate")

    def test_healthy_stream_continues(self):
        v = assess_partial(HEALTHY)
        assert isinstance(v, QualityVerdict)
        assert v.action == CONTINUE
        assert v.reasons == []
        assert v.score == 1.0

    def test_healthy_partial_prefix_continues(self):
        # Every cadence sample of a good answer must read as "continue".
        for cut in range(240, len(HEALTHY), 100):
            assert assess_partial(HEALTHY[:cut]).action == CONTINUE

    def test_refusal_is_flagged_not_regenerated(self):
        v = assess_partial("I can't help with that.")
        assert v.action == FLAG            # conservative: flag, never kill
        assert "refusal_leak" in v.reasons
        assert v.score < 1.0

    def test_refusal_allowed_when_expected(self):
        v = assess_partial("I can't help with that.", expect_refusal_ok=True)
        assert v.action == CONTINUE
        assert v.reasons == []

    def test_as_an_ai_leak_flagged(self):
        assert "refusal_leak" in assess_partial(
            "As an AI language model, I do not have opinions.").reasons

    def test_repetition_loop_regenerates(self):
        v = assess_partial("the cache the cache " * 20)
        assert v.action == REGENERATE
        assert any(r.startswith("degenerate_repetition") for r in v.reasons)
        assert v.score < 0.4

    def test_mild_repetition_only_flags(self):
        # ~2/3 repeated tokens: suspicious, not conclusive.
        v = assess_partial("alpha beta gamma " + "delta delta " * 8)
        assert v.action == FLAG
        assert any(r.startswith("degenerate_repetition") for r in v.reasons)

    def test_empty_and_whitespace_streams_are_reported_empty(self):
        for text in ("", "   \n  ", None):
            v = assess_partial(text)
            assert "empty" in v.reasons
            assert v.score < 1.0

    def test_error_spike_flagged(self):
        v = assess_partial(
            "error while parsing. the call failed with an exception and the "
            "value was undefined, so the traceback shows the failed retry.")
        assert any(r.startswith("error_spike") for r in v.reasons)
        assert v.action in (FLAG, REGENERATE)

    def test_code_about_errors_is_not_a_false_positive(self):
        # A legitimate technical answer that merely *talks* about errors must
        # never be killed — flagging is the worst that may happen.
        v = assess_partial(
            "try:\n    load(path)\nexcept FileNotFoundError as exc:\n"
            "    raise ConfigError('missing file') from exc\n"
            "The exception is re-raised so the caller sees a typed error "
            "rather than an undefined value or a raw traceback.")
        assert v.action in (CONTINUE, FLAG)


class TestGuardedStreamWiring:
    def test_controller_is_consulted_during_a_stream(self, monkeypatch):
        seen: list[str] = []

        def spy(text, **kw):
            seen.append(text)
            return QualityVerdict(CONTINUE, 1.0, [])

        monkeypatch.setattr(
            "app.quality.stream_controller.assess_partial", spy)
        out = _drain(LLMClient()._guarded_stream(
            _gen([HEALTHY[i:i + 40] for i in range(0, len(HEALTHY), 40)])))
        assert out == HEALTHY               # healthy text passes untouched
        assert seen, "assess_partial was never called during the stream"
        # Sampled, not per-token: far fewer calls than chunks.
        assert len(seen) < len(HEALTHY) / 100

    def test_healthy_stream_is_not_sampled_before_a_minimum_prefix(
            self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            "app.quality.stream_controller.assess_partial",
            lambda text, **kw: (seen.append(text),
                                QualityVerdict(CONTINUE, 1.0, []))[1])
        out = _drain(LLMClient()._guarded_stream(_gen(["short answer."])))
        assert out == "short answer."
        # Only the end-of-stream read — no mid-stream sample on a tiny answer.
        assert len(seen) == 1

    def test_verdict_is_exposed_to_the_caller(self):
        _out, v = _drain_with_verdict(LLMClient()._guarded_stream(_gen([HEALTHY])))
        assert v is not None and v.action == CONTINUE

    def test_refusal_streams_to_completion_flagged_not_truncated(self):
        text = ("I'm unable to assist with that request. " + HEALTHY)
        out, v = _drain_with_verdict(LLMClient()._guarded_stream(_gen([text])))
        assert out == text                  # flagged, never cut short
        assert v.action == FLAG and "refusal_leak" in v.reasons

    def test_unpunctuated_degeneration_is_stopped(self):
        # No sentence terminator anywhere → RepetitionGuard's sentence splitter
        # never fires and the 120k char ceiling is far away. Only the quality
        # controller can catch this one.
        chunks = ["spam bucket " for _ in range(120)]
        out = _drain(LLMClient()._guarded_stream(_gen(chunks)))
        assert "output stopped" in out
        assert "degenerating" in out
        assert len(out) < len("".join(chunks))     # cut early

    def test_controller_exception_never_breaks_the_stream(self, monkeypatch):
        def boom(text, **kw):
            raise RuntimeError("controller exploded")

        monkeypatch.setattr(
            "app.quality.stream_controller.assess_partial", boom)
        out = _drain(LLMClient()._guarded_stream(
            _gen([HEALTHY[i:i + 40] for i in range(0, len(HEALTHY), 40)])))
        assert out == HEALTHY                # fail-open: stream intact

    def test_controller_import_failure_never_breaks_the_stream(self, monkeypatch):
        monkeypatch.delattr("app.quality.stream_controller.assess_partial")
        out = _drain(LLMClient()._guarded_stream(_gen([HEALTHY])))
        assert out == HEALTHY

    def test_flag_off_disables_the_controller(self, monkeypatch):
        called: list[str] = []
        monkeypatch.setattr(
            "app.quality.stream_controller.assess_partial",
            lambda text, **kw: (called.append(text),
                                QualityVerdict(CONTINUE, 1.0, []))[1])
        monkeypatch.setattr(
            "app.core.llm_client.cfg",
            SimpleNamespace(llm=cfg.llm,
                            quality=SimpleNamespace(stream_control=False)))
        out = _drain(LLMClient()._guarded_stream(_gen(["spam bucket " * 120])))
        assert out == "spam bucket " * 120   # nothing stopped, nothing assessed
        assert called == []

    def test_stream_guard_still_owns_punctuated_loops(self):
        # Regression: the two guards must not fight. A classic sentence loop is
        # still the stream guard's kill, with its own message.
        sent = "This sentence repeats verbatim every single time. "
        out = _drain(LLMClient()._guarded_stream(_gen([sent] * 10)))
        assert "repeating itself" in out


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
