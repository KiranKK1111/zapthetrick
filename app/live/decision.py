"""
Unified live Decision Engine.

Every "should we answer this?" judgement used to live inline in
routes_ws.py — interruption handling, satisfaction/feedback detection,
rhetorical suppression, implicit-question promotion, event answerability
and the ensemble gate — six scattered blocks. This module is the single
place those decisions are made; the WebSocket layer asks once and acts on
the returned verdict.

Two entry points, mirroring the two decision moments in the pipeline:

  decide_utterance(...)  — BEFORE detection: react to the raw utterance
                           (cancel in-flight answers, absorb feedback,
                           suppress rhetorical, flag implicit questions).
  decide_event(...)      — AFTER detection: given the typed event (and the
                           ensemble score on audio), commit to answering
                           or skip with a reason.

Every check is flag-gated by the same cfg.live.* switches as before and
fails open (a broken detector must never eat a real question).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config_loader import cfg

log = logging.getLogger(__name__)

# Verdict actions.
ANSWER = "answer"          # proceed to detection / generation
SKIP = "skip"              # stay quiet; `reason` says why, `frames` inform UI
CANCEL_THEN_ANSWER = "cancel_then_answer"  # cancel in-flight, then proceed


@dataclass
class Decision:
    action: str = ANSWER
    reason: str = ""
    # WS frames the caller should send (feedback / rhetorical meta …).
    frames: list[dict] = field(default_factory=list)
    # Additive signals to merge into downstream meta (implicit question …).
    signals: dict = field(default_factory=dict)


def decide_utterance(utterance: str, *, is_audio: bool, world_model=None) -> Decision:
    """Pre-detection decision on a finalized utterance (audio path rules only
    apply when `is_audio`; typed input is always answered)."""
    d = Decision()

    # 1. Interruption / self-correction — "actually, leave that…" cancels
    #    whatever is generating and answers the new utterance.
    if getattr(cfg.live, "interruption_handling", False) and is_audio:
        try:
            from app.live import interrupt as _intr
            if _intr.should_cancel(utterance):
                d.action = CANCEL_THEN_ANSWER
                d.reason = "interruption"
        except Exception:  # noqa: BLE001
            pass

    # 2. Interviewer satisfaction / feedback — "good", "not quite" is a
    #    reaction to the candidate's answer, never a question.
    if getattr(cfg.live, "satisfaction_detection", False) and is_audio:
        try:
            from app.live import satisfaction as _sat
            fb = _sat.classify_feedback(utterance)
            if fb is not None:
                d.action = SKIP
                d.reason = "feedback"
                d.frames.append(
                    {"type": "feedback", "state": fb, "text": utterance})
                return d
        except Exception:  # noqa: BLE001
            pass

    # 3. Rhetorical suppression — "…right?" / self-answered questions.
    if getattr(cfg.live, "rhetorical", False) and is_audio:
        try:
            from app.live import rhetorical as _rhet
            if not _rhet.should_answer(utterance):
                d.action = SKIP
                d.reason = "rhetorical"
                d.frames.append({
                    "type": "meta", "is_question": False,
                    "qtype": "rhetorical", "question": utterance,
                    "source": "rhetorical",
                })
                return d
        except Exception:  # noqa: BLE001
            pass

    # 4. Implicit-question promotion — "walk me through your project" has no
    #    question mark but must be answered. Additive signal only.
    if getattr(cfg.live, "implicit_question", False):
        try:
            from app.live import implicit as _impl
            sig = _impl.detect_implicit(utterance)
            if sig.is_implicit_question:
                d.signals["implicit"] = round(float(sig.confidence), 3)
                d.frames.append({
                    "type": "meta", "is_question": True, "qtype": "implicit",
                    "question": utterance, "source": "implicit",
                    "confidence": sig.confidence,
                })
        except Exception:  # noqa: BLE001
            pass

    # 5. Hypothetical / assumption scenario probes — "Suppose one service
    #    goes down." expects an answer with no wh-word and no '?'. Additive
    #    signal; the post-detection promotion in decide_event acts on it.
    if getattr(cfg.live, "hypothetical_question", True):
        try:
            from app.live import implicit as _impl2
            hsig = _impl2.detect_hypothetical(utterance)
            if hsig.is_implicit_question:
                d.signals["hypothetical"] = round(float(hsig.confidence), 3)
                d.frames.append({
                    "type": "meta", "is_question": True,
                    "qtype": "hypothetical", "question": utterance,
                    "source": "hypothetical", "confidence": hsig.confidence,
                })
        except Exception:  # noqa: BLE001
            pass

    return d


def decide_event(event, *, is_audio: bool, utterance: str, audio_np=None,
                 tracker=None) -> Decision:
    """Post-detection decision: answer the typed event or skip. Runs the
    ensemble gate (rules + agent + prosody) on the audio path so an
    explanation misread as a question is dropped, then applies the
    answerability rule (audio answers only real questions)."""
    d = Decision()

    # Ensemble gate — false-positive guard, agent-dominant.
    if (getattr(cfg.live, "ensemble_detection", False) and is_audio
            and getattr(event, "is_answerable", False)):
        try:
            from app.question_detection.classifier import heuristic_classify
            from app.live import ensemble as _ens
            h = heuristic_classify(utterance)
            pscore = None
            if audio_np is not None:
                try:
                    from app.question_detection.prosody_analyzer import analyze
                    pscore = analyze(audio_np).is_question_acoustic
                except Exception:  # noqa: BLE001
                    pscore = None
            thr = 0.5
            if getattr(cfg.live, "style_learning", False) and tracker is not None:
                try:
                    from app.live import style as _style
                    thr = max(0.2, min(0.8, 0.5 + _style.for_tracker(
                        tracker).threshold_adjustment()))
                except Exception:  # noqa: BLE001
                    thr = 0.5
            # Self-tuning from the accuracy ledger's user corrections
            # (2026-07-09): "should have answered" feedback lowers the gate,
            # "shouldn't have" raises it. Bounded ±0.10 inside the ledger.
            try:
                from app.live import ledger as _lg
                thr = max(0.2, min(0.8, thr - _lg.answer_bias()))
            except Exception:  # noqa: BLE001
                pass
            dec = _ens.decide(
                agent_is_q=True, agent_conf=0.85,
                heuristic_is_q=h.is_question, heuristic_conf=h.confidence,
                prosody_score=pscore, threshold=thr,
            )
            d.signals["detection_confidence"] = round(dec.score, 3)
            if not dec.is_question:
                d.action = SKIP
                d.reason = "ensemble_not_question"
                return d
        except Exception:  # noqa: BLE001
            pass

    # Answerability — on the audio path only real questions are answered.
    if is_audio and not getattr(event, "is_answerable", False):
        # PROMOTION: the typed event says "not a question", but three signal
        # classes are reliable probes the typer routinely misses —
        #   * indirect imperatives ("walk me through your project"),
        #   * hypothetical scenarios ("suppose the DB goes down."),
        #   * a strong terminal pitch RISE (the interviewer's tone asked,
        #     even if the words read like a statement).
        # These answer instead of skipping; the signal is surfaced in meta.
        promo = _promotion(utterance, audio_np)
        if promo is not None:
            d.signals["promoted"] = promo[0]
            d.signals["promoted_confidence"] = round(float(promo[1]), 3)
            d.signals["promoted_qtype"] = promo[2]
            return d
        d.action = SKIP
        d.reason = str(getattr(event, "kind", "not_answerable"))
        return d

    return d


def _promotion(utterance: str, audio_np=None) -> tuple[str, float, str] | None:
    """(signal_name, confidence, qtype) when a not-answerable utterance
    should be answered anyway; None to keep the skip. Never raises."""
    try:
        _bias = 0.0
        try:
            from app.live import ledger as _lg
            _bias = _lg.answer_bias()
        except Exception:  # noqa: BLE001
            _bias = 0.0
        from app.live import implicit as _impl
        if getattr(cfg.live, "implicit_question", False):
            sig = _impl.detect_implicit(utterance)
            if sig.is_implicit_question and sig.confidence >= 0.55 - _bias:
                return ("implicit", sig.confidence, "implicit")
        if getattr(cfg.live, "hypothetical_question", True):
            hsig = _impl.detect_hypothetical(utterance)
            if hsig.is_implicit_question and hsig.confidence >= 0.6 - _bias:
                return ("hypothetical", hsig.confidence, "hypothetical")
    except Exception:  # noqa: BLE001
        pass
    # Tone: a clear rising terminal pitch on a multi-word utterance is a
    # question by delivery ("you've used Kafka in production…?"). Threshold
    # is deliberately high — uptalk on real statements must not flood the
    # session with answers.
    if (getattr(cfg.live, "prosody_promotion", True)
            and audio_np is not None
            and len((utterance or "").split()) >= 3):
        try:
            from app.question_detection.prosody_analyzer import analyze
            pscore = float(analyze(audio_np).is_question_acoustic)
            if pscore >= 0.72:
                return ("prosody", pscore, "implicit")
        except Exception:  # noqa: BLE001
            pass
    return None


def admit_answer(sid: str):
    """Resource admission for one answer generation. Returns (ok, budget):
    `budget` must be `.release()`d by the caller when generation ends.
    Delegates to the session budget (concurrent-answer cap)."""
    if not getattr(cfg.live, "session_budget", False):
        return True, None
    try:
        from app.live.budget import get_budget
        budget = get_budget(sid)
        if not budget.acquire():
            return False, None
        return True, budget
    except Exception:  # noqa: BLE001
        return True, None
