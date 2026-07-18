"""Live-path accuracy + the routing-latency contract.

Covers the three levers added for "transcribe technical questions correctly
and answer them in a fraction of a second":
  - the prediction agent auto-corrects mis-transcribed technical terms AND
    judges difficulty in the SAME call (so the orchestrator can skip its own
    difficulty round trip);
  - the segmenter passes the per-session biasing prompt into transcription.
"""
from __future__ import annotations

import asyncio

import numpy as np


# ── prediction agent: auto-correct + difficulty in one call ──────────────────
def test_predict_corrects_term_and_parses_difficulty(monkeypatch):
    from app.question_detection import agent

    async def fake_chat_json(messages, model=None):
        return (
            '{"is_question": true, "question": "Why do we need interpolation?",'
            ' "type": "technical_concept", "topic": "interpolation",'
            ' "difficulty": "hard"}'
        )

    monkeypatch.setattr(agent.llm, "chat_json", fake_chat_json)
    pred = asyncio.run(agent.predict("why do we need innovation", []))
    assert pred.is_question is True
    assert "interpolation" in pred.question.lower()
    assert pred.type == "technical_concept"
    assert pred.difficulty == "hard"


def test_predict_invalid_difficulty_falls_back_to_standard(monkeypatch):
    from app.question_detection import agent

    async def fake_chat_json(messages, model=None):
        return '{"is_question": true, "question": "x", "type": "coding", "difficulty": "ultra"}'

    monkeypatch.setattr(agent.llm, "chat_json", fake_chat_json)
    pred = asyncio.run(agent.predict("write a function", []))
    assert pred.difficulty == "standard"
    assert pred.type == "coding"


def test_predict_non_question_has_empty_question(monkeypatch):
    from app.question_detection import agent

    async def fake_chat_json(messages, model=None):
        return '{"is_question": false, "question": "", "type": "smalltalk", "difficulty": "trivial"}'

    monkeypatch.setattr(agent.llm, "chat_json", fake_chat_json)
    pred = asyncio.run(agent.predict("yeah right, okay", []))
    assert pred.is_question is False
    assert pred.question == ""


def test_predict_model_prefers_fast_live_model(monkeypatch):
    """Prediction should route through the pinned fast live model when set."""
    from app.question_detection import agent
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.llm, "live_model", "llama-3.3-70b-versatile")
    assert agent._model() == "llama-3.3-70b-versatile"


# ── segmenter: per-session biasing prompt reaches transcription ──────────────
class _Clock:
    def __init__(self, values):
        self._values = list(values)
        self._last = 0.0

    def __call__(self):
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_segmenter_passes_prompt_provider_to_stt(monkeypatch):
    from app.audio import stream as sm
    from tests.test_audio_segmenter import FakeStreamingVAD, _silence, _voice

    captured: dict = {}

    async def fake_transcribe(audio, prompt=None):
        captured["prompt"] = prompt
        return "ok", None

    monkeypatch.setattr(
        sm.stt_factory, "transcribe_with_confidence", fake_transcribe)
    monkeypatch.setattr(sm.vad, "StreamingVAD", FakeStreamingVAD)

    emitted: list[str] = []

    async def on_utt(text, audio):
        emitted.append(text)

    seg = sm.AudioStreamSegmenter(on_utterance=on_utt, prompt_provider=lambda: "MY BIAS")

    async def run():
        await seg.push(_voice(800))     # speech (>= min_utterance)
        await seg.push(_silence(700))   # endpoint → finalise
        await seg.flush()

    asyncio.run(run())
    assert emitted == ["ok"]
    assert captured["prompt"] == "MY BIAS"


# ── conservative correction: a real word is kept verbatim ────────────────────
def test_predict_keeps_real_word_when_model_returns_it(monkeypatch):
    """The conservative prompt should round-trip a real word like 'cohesion'
    (the model returns it unchanged) rather than swapping in a named model."""
    from app.question_detection import agent

    async def fake_chat_json(messages, model=None):
        # Sanity: the prompt must instruct conservative behaviour.
        sys = messages[0]["content"].lower()
        assert "trust the transcript" in sys
        assert "boehm" in sys  # the explicit do-not-invent example
        return ('{"is_question": true, "question": "What is cohesion?",'
                ' "type": "technical_concept", "topic": "cohesion",'
                ' "difficulty": "standard"}')

    monkeypatch.setattr(agent.llm, "chat_json", fake_chat_json)
    pred = asyncio.run(agent.predict("what is cohesion", []))
    assert pred.question == "What is cohesion?"
    assert "boehm" not in pred.question.lower()


# ── live answers: concise prompt + capped length + first-token watchdog ──────
def test_persona_stream_concise_uses_terse_prompt_and_caps_tokens(monkeypatch):
    from app.tools import persona_answer

    captured: dict = {}

    async def fake_stream_chat(messages, model=None, options=None):
        captured["options"] = options
        captured["system"] = messages[0]["content"]
        yield "ok"

    monkeypatch.setattr(persona_answer.llm, "stream_chat", fake_stream_chat)

    async def run():
        out = []
        # concise=True selects the terse real-time prompt (the live path only
        # sets this when deliberation asks for a concise answer or the operator
        # turns live_detailed off).
        async for c in persona_answer.stream(
            question="q", profile={}, concise=True, max_tokens=123
        ):
            out.append(c)
        return out

    out = asyncio.run(run())
    assert out == ["ok"]
    assert captured["options"]["max_tokens"] == 123
    assert "real time" in captured["system"].lower()


def test_persona_stream_default_is_detailed_prompt(monkeypatch):
    """Live now defaults to the DETAILED chat-quality prompt (concise=False)."""
    from app.tools import persona_answer

    captured: dict = {}

    async def fake_stream_chat(messages, model=None, options=None):
        captured["system"] = messages[0]["content"]
        yield "ok"

    monkeypatch.setattr(persona_answer.llm, "stream_chat", fake_stream_chat)

    async def run():
        async for _ in persona_answer.stream(question="q", profile={}):
            pass

    asyncio.run(run())
    # The detailed interview prompt is NOT the terse "real time" one.
    assert "real time" not in captured["system"].lower()


def test_live_first_token_watchdog_aborts_a_stalled_model(monkeypatch):
    from app.core import orchestrator as orch
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.llm, "live_first_token_timeout", 0.05)

    async def hanging_stream(**kwargs):
        await asyncio.sleep(5)
        yield "never"

    monkeypatch.setattr(orch.persona_answer, "stream", lambda **kw: hanging_stream(**kw))

    async def run():
        ctx = orch.AnswerContext(
            question="What is cohesion?",
            session_id="t-watchdog",
            profile={},
            forced_type="technical_concept",
            forced_difficulty="standard",
            skip_embedding=True,
            live=True,
        )
        kinds = []
        async for ev in orch.answer_question(ctx):
            kinds.append(ev.kind)
        return kinds

    kinds = asyncio.run(run())
    assert "error" in kinds
    assert "token" not in kinds
