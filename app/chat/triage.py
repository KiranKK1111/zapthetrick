"""Single-call turn triage: difficulty + document-intent in ONE LLM round-trip.

The chat and upload paths previously made TWO separate classifier calls per
turn (difficulty + document-intent) on top of the clarifier and the answer.
That fan-out adds latency and burns rate-limit headroom on free keys. `triage`
folds the two label classifications into one call, while preserving the
deterministic fast-paths from `difficulty` (greetings → trivial, heavy
generation → expert) so the common cases stay instant.

Returns a `Triage(difficulty, wants_document, doc_format)`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Reuse the difficulty fast-paths + level set so behaviour stays consistent.
from app.chat.difficulty import LEVELS, _is_heavy, _TRIVIAL_PHRASES  # noqa: E402
from app.core import lexicons  # noqa: E402
from app.documents.detect import _FORMATS, explicit_doc_request  # noqa: E402

# A downloadable-artifact token: a named file format, the words file/document/
# attachment, or an explicit produce/deliver-as-a-file verb (download/export/
# attach/save). This is the PRECISION guard that lets us trust the LLM's
# `document:true` under explicit-only mode: it separates a real "produce a file"
# request ("give me a report AS A PDF", "export this", "put it in a word doc")
# from an in-chat "summarize this" / "give me a report" — the latter names no
# artifact and so never generates a file. Deliberately broad on recall (catches
# phrasings the stricter `explicit_doc_request` regex misses) but still requires
# SOME artifact signal, so it cannot fire on a bare summary/answer request.
_ARTIFACT_RE = re.compile(lexicons.TRIAGE_ARTIFACT, re.I)


def _mentions_artifact(text: str) -> bool:
    """True if the message names a downloadable deliverable (a file format, the
    words file/document/attachment, or a deliver-as-a-file verb).

    DETERMINISTIC lexicon ONLY. The semantic `artifact_delivery` gate was removed
    here (2026-07-14): as the PRECISION guard that lets the LLM's `document:true`
    generate a file, a fuzzy embedding match over-fired on ordinary answer-
    seeking turns ("Can I have a solution for …", "give me the approach") and
    produced UNREQUESTED PDFs — the exact bug this guard exists to prevent. It
    also made behavior inconsistent: with the embedder cold (tests) the gate is
    silent, so a false positive only surfaced in the warm app. Recall for
    phrasings the lexicon misses still flows through `explicit_doc_request`
    (`wants_doc = det or verified`), which has its own `document_request`
    semantic tail — the gate that actually models doc INTENT, not just artifact
    word-proximity.
    """
    return bool(_ARTIFACT_RE.search(text or ""))


@dataclass
class Triage:
    difficulty: str = "standard"
    wants_document: bool = False
    doc_format: str = "pdf"
    # #12: cheap deterministic signals folded into the one triage object so
    # downstream reads them from here instead of recomputing separately.
    topic_shift: bool = False       # explicit "new topic" cue → answer fresh
    read_only: bool = False         # explain/review existing content (a lookup)


def _signals(text: str) -> tuple[bool, bool]:
    """(topic_shift, read_only) — deterministic, no LLM cost. Fail-open."""
    ts = ro = False
    try:
        from app.followup.acts import is_topic_shift
        ts = is_topic_shift(text)
    except Exception:  # noqa: BLE001
        ts = False
    try:
        from app.clarify.intent_pipeline import is_read_only
        ro = is_read_only(text)
    except Exception:  # noqa: BLE001
        ro = False
    return ts, ro


_PROMPT = (
    "Classify the user's message on TWO axes and reply with ONLY compact JSON.\n"
    "1) difficulty — how computationally demanding it is to answer WELL:\n"
    "   trivial = greetings/small talk/one-line facts; standard = ordinary "
    "questions/explanations/straightforward code; hard = multi-step reasoning, "
    "non-trivial algorithms/math, debugging, system design; expert = deep/novel "
    "problem-solving, proofs, intricate optimization, large multi-file builds.\n"
    "2) document — true ONLY if the user is asking to PRODUCE a downloadable "
    "file/document (e.g. \"make a PDF\", \"export as Excel\", \"zip the "
    "project\"); false for ordinary questions/explanations/code. When true, "
    "give the format they named (pdf|docx|xlsx|csv|json|md|txt|zip), defaulting to "
    "pdf, and 'zip' for a zip/archive/whole-project download.\n\n"
    "Reply with EXACTLY: {\"difficulty\": \"trivial|standard|hard|expert\", "
    "\"document\": true|false, \"format\": \"pdf|docx|xlsx|csv|json|md|txt|zip\"}\n\n"
    "{ctx}User message:\n{text}"
)


def _parse(raw: str) -> dict:
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


async def triage(text: str, recent: str = "", *,
                 allow_recent_doc: bool = False) -> Triage:
    """One LLM call → (difficulty, wants_document, doc_format). Deterministic
    fast-paths short-circuit difficulty; greetings also skip the call entirely
    (a greeting never requests a document).

    `allow_recent_doc`: whether a file/document request in the PRIOR turn may
    carry into this one. Default FALSE so file intent is strictly per-turn —
    otherwise "give me a .py file" (turn N-1) leaks into an unrelated program
    request (turn N). The caller sets it True ONLY when this turn is a genuine
    clarification answer (client `skip_clarify` or a CLARIFICATION_ANSWER act),
    so a clarifier round-trip still inherits the original generate intent."""
    t = (text or "").strip()
    if not t:
        return Triage()

    _ts, _ro = _signals(t)
    low = t.lower().strip(" \t\n!.?,;:")
    fast_difficulty: str | None = None
    if low in _TRIVIAL_PHRASES or len(low) <= 2:
        # A greeting/ack never asks for a file — skip the LLM call outright.
        return Triage(difficulty="trivial", wants_document=False,
                      topic_shift=_ts, read_only=_ro)
    if _is_heavy(t):
        fast_difficulty = "expert"  # still classify doc-intent below

    try:
        from app.core.config_loader import cfg
        from app.core.llm_client import llm

        ctx = (recent or "").strip()
        ctx_block = (
            "Recent conversation (context; the latest message may be a "
            f"follow-up):\n{ctx[:2000]}\n\n" if ctx else ""
        )
        # fill(), not str.format(): the template embeds a literal JSON example
        # whose braces made .format() raise KeyError('"difficulty"') on EVERY
        # call — the triage LLM leg was silently dead (deterministic fallback
        # only) until this surfaced in the logs.
        from app.core.prompt import fill as _fill
        _messages = [{"role": "user",
                      "content": _fill(_PROMPT, ctx=ctx_block,
                                       text=t[:4000])}]
        _opts = {"temperature": cfg.temperature.classifier,
                 "num_predict": cfg.output_tokens.intent}
        try:
            raw = await llm.complete(
                _messages,
                model=(cfg.llm.classifier_model or cfg.llm.model),
                options=_opts,
            )
        except Exception:  # noqa: BLE001
            # One retry on the DEFAULT model: the dedicated classifier model is
            # often the flaky/free one, and a triage miss silently drops the
            # turn's document decision. Same-model failures just re-raise.
            if cfg.llm.classifier_model and \
                    cfg.llm.classifier_model != cfg.llm.model:
                raw = await llm.complete(_messages, model=cfg.llm.model,
                                         options=_opts)
            else:
                raise
    except Exception as exc:  # noqa: BLE001 — never block the turn on triage
        log.warning("triage LLM failed, deterministic fallback "
                    "(doc-intent may be missed): %s", exc)
        det, det_fmt = explicit_doc_request(t)
        if not det and recent and allow_recent_doc:
            det, det_fmt = explicit_doc_request(recent)
        return Triage(
            difficulty=fast_difficulty or "standard",
            wants_document=det,
            doc_format=det_fmt or "pdf",
            topic_shift=_ts,
            read_only=_ro,
        )

    obj = _parse(raw)
    difficulty = fast_difficulty or str(obj.get("difficulty", "")).lower().strip()
    if difficulty not in LEVELS:
        difficulty = "standard"
    fmt = str(obj.get("format") or "pdf").lower().strip()
    if fmt not in _FORMATS:
        fmt = "pdf"
    wants_doc = bool(obj.get("document"))

    # Explicit request on THIS turn. The immediately-preceding turn is only
    # consulted when this turn is a clarification answer (allow_recent_doc) —
    # so a fresh/new-topic request never inherits a prior "make a file".
    det, det_fmt = explicit_doc_request(t)
    if not det and recent and allow_recent_doc:
        det, det_fmt = explicit_doc_request(recent)

    # Strict gate (documents.explicit_only, default ON): only a clear,
    # deterministic "produce a file" request generates a document. The LLM
    # classifier over-triggers on ordinary summaries/answers, so its
    # `document:true` is NOT trusted on its own — the user must explicitly ask.
    try:
        from app.core.config_loader import cfg as _cfg
        explicit_only = bool(getattr(_cfg.documents, "explicit_only", True))
    except Exception:  # noqa: BLE001
        explicit_only = True

    if explicit_only:
        # Explicit-only, but with reliable RECALL: a document is produced when
        # the deterministic detector fires, OR when the LLM says "document" AND
        # the message actually names a downloadable artifact (format/file/
        # download/export). The artifact guard preserves precision — it blocks
        # the classic "summarize this" / "give me a report" false positives
        # (no artifact token) that made docs appear unrequested — while the LLM
        # leg recovers explicit phrasings the regex alone misses (fixing the
        # "explicitly asked but nothing generated" case). Never trusts the raw
        # LLM flag on its own.
        verified = wants_doc and (
            _mentions_artifact(t)
            or (allow_recent_doc and bool(recent) and _mentions_artifact(recent)))
        wants_doc = det or verified
        if wants_doc:
            fmt = det_fmt or fmt
    else:
        # Looser legacy behavior: an explicit request still overrides the LLM.
        if det:
            wants_doc = True
            fmt = det_fmt or fmt

    return Triage(
        difficulty=difficulty,
        wants_document=wants_doc,
        doc_format=fmt,
        topic_shift=_ts,
        read_only=_ro,
    )


__all__ = ["triage", "Triage"]
