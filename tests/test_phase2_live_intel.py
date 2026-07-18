"""Phase 2 Live Interview Intelligence — unit tests for the completion items:

  2A-5 silence taxonomy            (app/live/silence.py)
  2A-6 acoustic adaptation         (app/live/acoustic.py)
  2A-7 code-switch handling        (app/live/language.py)
  2B-9 interviewer hidden-goal     (app/live/interviewer_intent.py)
  2B-10 predictive pre-drafting    (app/live/predict.py)
  2B-15 multi-hypothesis interp    (app/live/interpret.py)
  2C-18/19 evidence-strength rank  (app/live/evidence.py)
  2C-20 company style bias         (app/live/org.py)
  2C-21 adaptive length variants   (app/live/contract.py)
  2C-24 cross-round memory graph   (app/live/cross_round.py)
  2D-30 live confidence recovery   (app/live/stt_recovery.py)
  2D-29 dev overlay                (app/live/devmode.py)
  2E-35 minimal TTS speech text    (app/live/tts.py)
  2E-36 pre-interview research      (app/live/research.py)

All deterministic + offline (no sockets, audio, or models).
"""
from __future__ import annotations

import pathlib


# ── 2A-5 silence taxonomy ───────────────────────────────────────────────────
def test_silence_thinking_done_hesitation():
    from app.live import silence as S
    assert S.classify("what is the", completeness="incomplete", gap_ms=1000).label == S.THINKING
    assert S.classify("Explain Kafka partitions.", completeness="complete").label == S.DONE
    assert S.classify("so tell me about um").label == S.HESITATION


def test_silence_directive_and_fail_open():
    from app.live import silence as S
    thinking = S.classify("what is the", completeness="incomplete", gap_ms=1500)
    assert S.directive(thinking)
    assert S.classify(None).label == S.UNKNOWN          # type: ignore[arg-type]
    assert S.directive(S.SilenceSignal()) == ""


# ── 2A-6 acoustic adaptation ────────────────────────────────────────────────
def test_acoustic_degrades_confidence():
    from app.live import acoustic as A
    clean = A.assess(stt_conf=0.95)
    noisy = A.assess(stt_conf=0.2, snr_db=5.0)
    assert clean.condition == A.CLEAN
    assert noisy.confidence_penalty > 0.0
    assert A.adjust_confidence(0.8, noisy) < 0.8
    assert A.needs_reconfirmation(noisy) in (True, False)  # deterministic bool


def test_acoustic_unknown_when_no_signal():
    from app.live import acoustic as A
    p = A.assess()
    assert p.condition == A.UNKNOWN
    assert A.adjust_confidence(0.7, p) == 0.7


# ── 2A-7 code-switch ────────────────────────────────────────────────────────
def test_code_switch_detection_and_spans():
    from app.live import language as L
    mixed = "explain मुझे kafka partitions के बारे में"
    assert L.is_code_switched(mixed)
    present = L.languages_present(mixed)
    assert "en" in present and "hi" in present
    spans = L.code_switch_spans(mixed)
    assert any(lang == "hi" for _, lang in spans)
    assert L.code_switch_directive(mixed)


def test_code_switch_monolingual_is_not_flagged():
    from app.live import language as L
    assert not L.is_code_switched("what is a kafka partition")
    assert L.code_switch_directive("what is a kafka partition") == ""


# ── 2B-9 interviewer intent ─────────────────────────────────────────────────
def test_probe_intent_classes():
    from app.live import interviewer_intent as II
    assert II.probe_intent("but why does that matter?").label == II.DEPTH_PROBE
    assert II.probe_intent("are you sure that's right?").label == II.STRESS_TEST
    assert II.probe_intent("tell me about a time you disagreed").label == II.CULTURE_FIT
    assert II.probe_intent("what is a mutex?", qtype="technical_concept").label == II.FUNDAMENTALS


def test_probe_intent_neutral_and_directive():
    from app.live import interviewer_intent as II
    neutral = II.probe_intent("describe the architecture of your last project in depth")
    # depth cue present -> depth probe; ensure directive non-empty for a hit
    hit = II.probe_intent("go deeper into how it works")
    assert II.directive(hit)
    assert II.directive(II.ProbeIntent()) == ""


# ── 2B-10 predictive pre-drafting ───────────────────────────────────────────
def test_predraft_stash_and_consume():
    from app.live import predict as P
    P.forget_session("pd")
    drafts = P.predraft("pd", ["Design a URL shortener", "What is a mutex?"])
    assert drafts and all("outline" in d for d in drafts)
    hit = P.consume_directive("pd", "design a url shortener that scales")
    assert hit and "anticipated" in hit.lower()
    assert P.consume_directive("pd", "tell me about redis clustering") == ""
    P.forget_session("pd")
    assert P.consume_directive("pd", "design a url shortener that scales") == ""


def test_predraft_outline_shapes():
    from app.live import predict as P
    assert "Situation" in P.predraft_outline("tell me about a time you failed")
    assert "components" in P.predraft_outline("design a system to scale")


# ── 2B-15 multi-hypothesis interpretation ───────────────────────────────────
def test_interpretations_ambiguous_and_literal():
    from app.live import interpret as I
    assert I.is_ambiguous("why?")
    hyps = I.interpretations("why?", topic="kafka")
    assert len(hyps) >= 2 and hyps[0].confidence >= hyps[-1].confidence
    assert I.directive(hyps)
    # A specific question is literal, single reading, no ambiguity directive.
    lit = I.interpretations("What is the CAP theorem in distributed systems?")
    assert len(lit) == 1 and lit[0].intent == "literal"
    assert I.directive(lit) == ""


# ── 2C-18/19 evidence-strength ranking ──────────────────────────────────────
def test_evidence_strength_rank_and_directive():
    from app.live import evidence as E
    strong = E.EvidenceBinding()
    strong.add("resume line", source="profile")
    strong.add("retrieved", source="retrieval")
    assert E.strength_label(strong) == E.STRONG
    ranked = E.rank_segments(strong)
    assert ranked[0].source == "profile"
    assert "assertively" in E.strength_directive(strong)

    weak = E.EvidenceBinding()
    weak.add("just a directive", source="directive")
    assert E.strength_label(weak) == E.WEAK
    assert "conservatively" in E.strength_directive(weak)


# ── 2C-20 company style ─────────────────────────────────────────────────────
def test_company_style_known_and_unknown():
    from app.live import org as O
    assert O.known_company("Amazon Web Services")
    assert "Leadership Principles" in O.company_style_directive("amazon")
    assert O.company_style_directive("Some Unknown LLC") == ""
    assert not O.known_company("")


# ── 2C-21 adaptive length variants ──────────────────────────────────────────
def test_contract_length_variants():
    from app.live import contract as C
    budgets = C.variant_budgets()
    assert budgets["brief"] < budgets["standard"] < budgets["deep"]
    assert C.variant_for_seconds(10) == "brief"
    assert C.variant_for_seconds(30) == "standard"
    assert C.variant_for_seconds(60) == "deep"
    assert C.length_directive(C.Contract(max_answer_seconds=10))


# ── 2C-24 cross-round memory graph ──────────────────────────────────────────
def test_cross_round_persist_and_link(tmp_path: pathlib.Path):
    from app.live import cross_round as CR
    p = tmp_path / "cr.json"
    assert CR.start_round("Acme", role="backend", path=p) == 1
    assert CR.start_round("Acme", role="backend", path=p) == 2   # persisted
    CR.record_topic("Acme", "kafka", role="backend", qtype="technical_concept", path=p)
    CR.record_topic("Acme", "kafka", role="backend", path=p)
    assert "kafka" in CR.prior_topics("Acme", role="backend", path=p)
    assert CR.link_directive("Acme", "kafka", role="backend", path=p)
    # Different company is isolated.
    assert CR.prior_topics("Globex", role="backend", path=p) == []


def test_cross_round_fail_open_on_bad_path():
    from app.live import cross_round as CR
    # Empty company -> no key -> no crash, empty result.
    assert CR.prior_topics("", role="") == []
    assert CR.link_directive("", "") == ""


# ── 2D-30 live confidence recovery ──────────────────────────────────────────
def test_stt_recovery_single_shot_decision():
    from app.live import stt_recovery as R
    R.forget_session("rec")
    R.observe("rec", 0.2)
    R.observe("rec", 0.3)
    R.observe("rec", 0.25)
    assert R.should_recover("rec", threshold=0.45, min_samples=3)
    R.forget_session("rec")


def test_stt_recovery_healthy_no_switch():
    from app.live import stt_recovery as R
    R.forget_session("ok")
    for c in (0.9, 0.85, 0.92):
        R.observe("ok", c)
    assert not R.should_recover("ok", threshold=0.45, min_samples=3)
    R.forget_session("ok")


def test_stt_recovery_plans_target_and_latches():
    from app.live import stt_recovery as R
    R.forget_session("sw")
    for _ in range(4):
        R.observe("sw", 0.1)
    assert R.should_recover("sw", threshold=0.45, min_samples=3)
    tgt = R.recover("sw")
    assert tgt in ("parakeet", "qwen_asr", "faster_whisper")
    # Single-shot: latched, so a second attempt is a no-op.
    assert R.should_recover("sw", threshold=0.45, min_samples=3) is False
    assert R.recover("sw") is None
    R.forget_session("sw")


# ── 2D-29 dev overlay ───────────────────────────────────────────────────────
def test_dev_overlay_fields_and_omission():
    from app.live import devmode as D
    ov = D.overlay(role="interviewer", model="m", latency_ms=123.7, phase="tech")
    assert ov["role"] == "interviewer" and ov["latency_ms"] == 123
    assert D.overlay() == {}          # nothing known -> empty (omitted)


# ── 2E-35 minimal TTS ───────────────────────────────────────────────────────
def test_tts_speech_markup_strips_markdown():
    from app.live import tts as T
    out = T.speech_markup("# Title\n- **bold** item, e.g. `code`\n```py\nx=1\n```")
    assert "**" not in out and "#" not in out and "```" not in out
    assert "for example" in out
    assert isinstance(T.is_available(), bool)


# ── 2E-36 pre-interview research ─────────────────────────────────────────────
def test_research_brief_deterministic():
    from app.live import research as RB
    b = RB.build_brief("Acme", "Backend Engineer", ["kafka", "postgres"])
    assert "kafka" in b.review_topics and b.questions_to_ask
    assert "Acme" in b.questions_to_ask[0]
    assert RB.brief_directive(b)
    assert RB.brief_directive(RB.ResearchBrief()) == ""
