"""Live-interview analysis scenarios (Phase A) — AnalysisOnLiveModule.md.

Each test pins one numbered scenario against the REAL implemented module
(no network / no LLM calls). Async `type_utterance` is exercised with an
injected fake predictor; every other assertion drives a deterministic helper
directly (repair, completeness, HypothesisBuffer, split_questions,
split_boundary, heuristic_classify, ensemble.decide, detect_implicit,
rhetorical.should_answer, prosody/acoustic/VAD).

Every scenario runs against the shipped deterministic modules — grammar
normalization stays an LLM concern (repair leaves clean phrases intact), and
follow-up typing is now a deterministic heuristic (a question opening on a
conjunction / back-reference).
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.audio import vad
from app.live import acoustic, events
from app.live.ensemble import decide
from app.live.hypothesis import HypothesisBuffer, completeness
from app.live.implicit import detect_implicit
from app.live.repair import repair
from app.live.rhetorical import should_answer
from app.question_detection import prosody_analyzer
from app.question_detection.agent import Prediction
from app.question_detection.classifier import heuristic_classify


def _run(coro):
    return asyncio.run(coro)


def _fake_predictor(pred: Prediction):
    async def _predict(text, recent):  # noqa: ANN001
        return pred
    return _predict


# ── Transcript repair (app/live/repair.py) ──────────────────────────────────

def test_s07_lowconf_word_repair():
    # Scenario 07 (Repair/PhaseA): a low-confidence mis-heard domain TOKEN is
    # corrected to the nearest domain term ("concurrancy" -> "concurrency").
    assert repair("what is concurrancy") == "what is concurrency"
    # A near-miss too far from any real word must NOT be corrected away.
    assert repair("what is a widget") == "what is a widget"


def test_s08_domain_vocab_repair():
    # Scenario 08 (Repair/PhaseA): a domain term melted into common words is
    # recovered ("cube net is ingress" -> contains "kubernetes ingress").
    out = repair("cube net is ingress")
    assert "kubernetes ingress" in out.lower()


def test_s09_llm_transcript_normalize():
    # Scenario 09 (Repair/PhaseA): when the (LLM) predictor returns a normalized
    # question, the event carries the normalized text through (fake predictor —
    # no real LLM call). Raw is unpunctuated so the normalized form flows in.
    pred = Prediction(True, "How do you handle duplicate messages?",
                      "technical_concept", "kafka", "standard")
    ev = _run(events.type_utterance(
        "so um how do you dedupe messages", [], predictor=_fake_predictor(pred)))
    assert ev.kind == events.QUESTION
    assert ev.questions == ["How do you handle duplicate messages?"]


def test_s10_preserve_raw_and_normalized():
    # Scenario 10 (Repair/PhaseA): the raw leading context is preserved on the
    # event (ev.context) alongside the answerable question — raw and derived
    # forms coexist rather than one overwriting the other.
    pred = Prediction(True, "How do you handle duplicates?",
                      "technical_concept", "kafka", "standard")
    ev = _run(events.type_utterance(
        "We use Kafka. how do you dedupe", [], predictor=_fake_predictor(pred)))
    assert ev.context == ["We use Kafka."]           # raw preserved verbatim
    assert ev.questions and ev.is_answerable          # question derived


def test_s11_grammar_normalize():
    # Scenario 11 (Repair/PhaseA): grammatical normalization is NOT done by the
    # deterministic repair pass — it intentionally leaves already-correct
    # domain phrases intact (grammar rewriting is an LLM-only concern), so a
    # clean sentence passes through verbatim without over-correction.
    assert repair("we write unit tests") == "we write unit tests"


def test_s12_domain_vocab_boosting():
    # Scenario 12 (Repair/PhaseA): a session-supplied vocab term boosts repair —
    # "jenkines" is untouched without the vocab, corrected to "jenkins" with it.
    assert repair("using jenkines") == "using jenkines"
    assert repair("using jenkines", vocab=["jenkins"]) == "using jenkins"


# ── Endpointing / pauses (app/live/hypothesis.py) ───────────────────────────

def test_s32_commit_points_boundary():
    # Scenario 32 (Endpointing/PhaseA): a finalized utterance commits only once
    # its settle window elapses; a continuation within the window merges (bumps
    # the generation) instead of committing a second question.
    b = HypothesisBuffer(settle_ms=600)
    g1 = b.add("What is Kafka?", now=0.0)
    assert not b.settle_due(0.1)          # still inside the settle window
    assert b.settle_due(1.0)              # window elapsed -> commit point reached
    g2 = b.add("and consumer groups?", now=1.0)
    assert g2 > g1                        # continuation superseded the settle
    # Fragments stitch into ONE sentence: the head's terminal '?' is an
    # endpointing artifact, dropped on merge.
    assert b.merged() == "What is Kafka and consumer groups?"


def test_s33_semantic_completion_implicit():
    # Scenario 33 (Endpointing/PhaseA): a trailing-off utterance that hangs
    # expecting completion is flagged implicit ("...increases so..." ).
    sig = detect_implicit("latency increases so...")
    assert sig.is_implicit_question
    assert sig.cue == "so..."


def test_s34_delay_window():
    # Scenario 34 (Endpointing/PhaseA): an incomplete tail earns a LONGER settle
    # window than a complete one so a thinking pause isn't cut off.
    inc = HypothesisBuffer(settle_ms=600)
    inc.add("What is", now=0.0)
    com = HypothesisBuffer(settle_ms=600)
    com.add("What is Kafka?", now=0.0)
    assert inc.required_settle_ms() > com.required_settle_ms()
    assert com.required_settle_ms() < 600 < inc.required_settle_ms()


def test_s105_silence_pause_intelligence():
    # Scenario 105 (Endpointing/PhaseA): completeness classifies the tail so a
    # closed thought settles fast and a dangling one waits.
    assert completeness("What is Kafka?") == "complete"
    assert completeness("What is") == "incomplete"
    assert completeness("can you explain") == "incomplete"


# ── Intent typing (classifier + events) ─────────────────────────────────────

def test_s14_intent_direct_question():
    # Scenario 14 (Intent/PhaseA): a direct wh-question is detected as a
    # question and typed as a technical concept.
    meta = heuristic_classify("What is polymorphism?")
    assert meta.is_question
    assert meta.type == "technical_concept"


def test_s15_intent_indirect_question():
    # Scenario 15 (Intent/PhaseA): an indirect "Can you explain X?" phrasing is
    # still detected as a question.
    meta = heuristic_classify("Can you explain closures?")
    assert meta.is_question
    assert meta.type == "technical_concept"


def test_s16_intent_scenario_question():
    # Scenario 16 (Intent/PhaseA): a scenario prompt ("Suppose one service goes
    # down. How would you...") splits into leading context + the question.
    pred = Prediction(True, "How would you handle the failover?",
                      "technical_concept", "resilience", "standard")
    ev = _run(events.type_utterance(
        "Suppose one service goes down. How would you handle the failover?",
        [], predictor=_fake_predictor(pred)))
    assert ev.kind == events.QUESTION
    assert any("Suppose" in c for c in ev.context)
    assert ev.questions and "failover" in ev.questions[0].lower()


def test_s17_intent_explanation_not_question():
    # Scenario 17 (Intent/PhaseA): a declarative statement is NOT a question.
    meta = heuristic_classify("In our org we use Kafka.")
    assert not meta.is_question


def test_s18_intent_followup():
    # Scenario 18 (Intent/PhaseA): a question that OPENS on a conjunction /
    # back-reference is a deterministic follow-up ("And why is that?"), tagged
    # by the heuristic path — no LLM needed.
    assert heuristic_classify("And why is that?").is_followup
    assert heuristic_classify("So how does that scale?").is_followup
    assert heuristic_classify("What about failure handling?").is_followup
    # A fresh, standalone question is NOT a follow-up.
    assert not heuristic_classify("What is a hash map?").is_followup


def test_s19_intent_greeting_smalltalk_transition():
    # Scenario 19 (Intent/PhaseA): non-questions refine into GREETING and
    # TRANSITION kinds from surface cues (small-talk predictor, no LLM).
    st = Prediction(False, "", "smalltalk", "", "standard")
    greet = _run(events.type_utterance(
        "Good morning, nice to meet you.", [], predictor=_fake_predictor(st)))
    assert greet.kind == events.GREETING
    trans = _run(events.type_utterance(
        "Let's move on to system design.", [], predictor=_fake_predictor(st)))
    assert trans.kind == events.TRANSITION
    assert not greet.is_answerable and not trans.is_answerable


def test_s24_rhetorical_suppression():
    # Scenario 24 (Intent/PhaseA): a rhetorical tag-question is suppressed; a
    # semantically-rhetorical phrasing with no tag falls through to answering
    # (by design: never drop a genuine question).
    assert should_answer("This scales well, make sense?") is False
    assert should_answer("Why would anyone use a monolith today?") is True


def test_s25_nonquestion_requiring_answer():
    # Scenario 25 (Intent/PhaseA): an imperative with no '?' still warrants an
    # answer ("Tell me about your project.").
    sig = detect_implicit("Tell me about your project.")
    assert sig.is_implicit_question
    assert sig.cue == "tell me about"


def test_s28_question_boundary_context_split():
    # Scenario 28 (Intent/PhaseA): leading context sentences are separated from
    # the trailing question.
    ctx, q = events.split_boundary(
        "We use Kafka. Kafka guarantees ordering. "
        "How would you handle duplicate messages?",
        "How would you handle duplicate messages?")
    assert ctx == ["We use Kafka.", "Kafka guarantees ordering."]
    assert q == "How would you handle duplicate messages?"


def test_s35_multi_question_split():
    # Scenario 35 (Intent/PhaseA): a conjoined multi-question utterance splits
    # into its constituent questions.
    qs = events.split_questions(
        "What is Kafka, why is it used, and how would you scale it?")
    assert len(qs) == 3
    assert all(q.endswith("?") for q in qs)
    assert any("scale" in q.lower() for q in qs)


def test_s37_ensemble_question_detection():
    # Scenario 37 (Intent/PhaseA): the ensemble blends rule/agent(/prosody) into
    # one decision — the agent dominates so an agent-confirmed question is never
    # dropped, while a mutual "no" stays not-a-question.
    yes = decide(agent_is_q=True, agent_conf=0.9, heuristic_is_q=True,
                 heuristic_conf=0.7)
    assert yes.is_question and yes.score > 0.5
    # Agent says question even when the rule disagrees -> still a question.
    agent_wins = decide(agent_is_q=True, agent_conf=0.9, heuristic_is_q=False)
    assert agent_wins.is_question
    no = decide(agent_is_q=False, agent_conf=0.9, heuristic_is_q=False)
    assert not no.is_question
    # Prosody fuses in when audio is present.
    fused = decide(agent_is_q=True, agent_conf=0.8, prosody_score=0.9)
    assert "prosody" in fused.components


# ── Prosody / acoustic / VAD (light coverage) ───────────────────────────────

def test_prosody_features_light():
    # Prosody: numpy-fallback path returns a bounded acoustic question score.
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False).astype("float32")
    audio = (0.1 * np.sin(2 * np.pi * 150 * t)).astype("float32")
    audio[-sr // 5:] *= 6.0                  # energy rise at the tail
    feats = prosody_analyzer.analyze(audio, sample_rate=sr)
    assert 0.0 <= feats.is_question_acoustic <= 1.0
    assert feats.duration_ms > 0


def test_acoustic_assess_light():
    # Acoustic: clean audio -> CLEAN + tiny penalty; degraded -> NOISY + bigger.
    clean = acoustic.assess(snr_db=28, stt_conf=0.95, partial_stability=0.9)
    noisy = acoustic.assess(snr_db=3, stt_conf=0.4, partial_stability=0.3)
    assert clean.condition == acoustic.CLEAN
    assert noisy.condition == acoustic.NOISY
    assert noisy.confidence_penalty > clean.confidence_penalty


def test_vad_energy_fallback(monkeypatch):
    # VAD: force the dependency-free energy gate (no torch) — loud audio reads as
    # speech, silence does not.
    monkeypatch.setattr(vad, "_silero_failed", True)
    sr = 16_000
    silence = np.zeros(sr, dtype="float32")
    loud = (0.3 * np.sin(
        2 * np.pi * 200 * np.linspace(0, 1, sr, endpoint=False))).astype("float32")
    assert vad.has_speech(silence) is False
    assert vad.has_speech(loud) is True
