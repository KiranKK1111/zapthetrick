"""Deterministic intent / confidence pre-gate (AnalysisOnIntentsAndConfidence).

Claude's perceived intelligence in *when to ask* comes from a multi-stage
decision pipeline that runs BEFORE clarification, not from a single confidence
scalar. This module implements that pipeline deterministically (no extra LLM
round-trip) so the clarifier can:

  • answer immediately when the request is already answerable (answer-first),
  • suppress questions about slots the user already supplied (self-critique),
  • only escalate to the LLM gate when a required slot is genuinely missing.

It consolidates the analysis doc's 40+ "layers" into the practical subsystems:
  1. Intent classification        → `detect_intent`
  2. Slot / entity extraction     → `extract_slots`
  3. Required-vs-optional fields  → `_required_optional`
  4. Answerability assessment     → `_answerability`
  5. Ambiguity detection          → `_ambiguity`
  6. Information-gain / cost model → `_gain`, `_cost`
  7. Confidence composition       → `_compose_confidence`
  8. Self-critique decision gate  → `assess` (ANSWER | CLARIFY | DEFER)

`assess()` is pure and fully unit-tested; the clarifier consumes its
`Assessment` to decide whether to answer directly, force a build clarification,
or defer the wording of a genuine question to the LLM gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chat.difficulty import (
    _PROJECT_NOUN_RE,
    _PROJECT_VERB_RE,
    _TECH_RE,
)
from ..core import lexicons

# ---- Intent vocabulary ----------------------------------------------------

INTENT_CHITCHAT = "chitchat"
INTENT_KNOWLEDGE = "knowledge"
INTENT_COMPARISON = "comparison"
INTENT_DEBUGGING = "debugging"
INTENT_TEST_GEN = "test_generation"
INTENT_DOCS = "documentation"
INTENT_DESIGN = "design"
INTENT_CODE_GEN = "code_generation"
INTENT_PROJECT_BUILD = "project_build"
INTENT_ARCHIVE = "archive"
INTENT_UNKNOWN = "unknown"

# Decisions the pipeline can reach.
ANSWER = "answer"      # answer-first: do NOT clarify
CLARIFY = "clarify"    # a required slot is missing → ask
DEFER = "defer"        # ambiguous → let the LLM gate decide the wording

# Pattern DATA lives in the central lexicon registry (app/core/lexicons.py);
# this module only compiles it. See the lexicons module docstring.
_GREETING_RE = re.compile(lexicons.INTENT_GREETING, re.IGNORECASE)
_KNOWLEDGE_RE = re.compile(lexicons.INTENT_KNOWLEDGE, re.IGNORECASE)
# Read-only "explain EXISTING content" request — never fire a "which language?"
# clarification for something the user pasted and wants explained.
_EXPLAIN_EXISTING_RE = re.compile(lexicons.INTENT_EXPLAIN_EXISTING,
                                  re.IGNORECASE)
# Pasted code + any explanation verb ⇒ a read-only ask.
_CODE_PASTE_RE = re.compile(lexicons.INTENT_CODE_PASTE, re.IGNORECASE)


def is_read_only(text: str) -> bool:
    """True when the turn asks to EXPLAIN/review existing or pasted content
    (a read-only lookup) rather than to write/build something new (#12).

    SEMANTIC-first (the `read_only` gate understands the explain-vs-build intent
    by meaning); the pasted-code signal stays deterministic (it's a structural
    fact, not intent), and the cue regex is the embedder-down fallback."""
    t = text or ""
    _code_lookup = _looks_like_pasted_code(t) and _KNOWLEDGE_RE.search(t) is not None
    try:
        from app.semantics.gates import matches
        v = matches("read_only", t)
        if v is not None:
            return v or _code_lookup
    except Exception:  # noqa: BLE001
        pass
    return bool(_EXPLAIN_EXISTING_RE.search(t)) or _code_lookup  # fallback


def _looks_like_pasted_code(text: str) -> bool:
    """True when the message appears to carry a pasted code block/snippet."""
    t = text or ""
    if "```" in t:
        return True
    return t.count("\n") >= 3 and bool(_CODE_PASTE_RE.search(t))
_COMPARISON_RE = re.compile(lexicons.INTENT_COMPARISON, re.IGNORECASE)
_DEBUG_RE = re.compile(lexicons.INTENT_DEBUG, re.IGNORECASE)
_TEST_RE = re.compile(lexicons.INTENT_TEST, re.IGNORECASE)
_DOCS_RE = re.compile(lexicons.INTENT_DOCS, re.IGNORECASE)
_DESIGN_RE = re.compile(lexicons.INTENT_DESIGN, re.IGNORECASE)
_CODE_GEN_RE = re.compile(lexicons.INTENT_CODE_GEN, re.IGNORECASE)
# Concrete, self-contained operations → a directly answerable code task even if
# no language is named (Claude picks a sensible default).
_OPERATION_RE = re.compile(lexicons.INTENT_OPERATION, re.IGNORECASE)
_PLATFORM_RE = re.compile(lexicons.INTENT_PLATFORM, re.IGNORECASE)
# Frameworks (subset of _TECH) used to tell "framework named" from "language".
_FRAMEWORK_RE = re.compile(lexicons.INTENT_FRAMEWORK, re.IGNORECASE)
# Technique/constraint cues that raise specificity (→ higher confidence).
_CONSTRAINT_RE = re.compile(lexicons.INTENT_CONSTRAINT, re.IGNORECASE)
# Vague scope markers that lower confidence / raise ambiguity.
_VAGUE_RE = re.compile(lexicons.INTENT_VAGUE, re.IGNORECASE)
# Document formats/targets — when a "document this" request names one, we can
# generate directly; otherwise the format/scope is a required missing choice.
_DOC_FORMAT_RE = re.compile(lexicons.INTENT_DOC_FORMAT, re.IGNORECASE)
# Archive / compress requests. When no archive FORMAT is named, ask which one.
_ARCHIVE_VERB_RE = re.compile(lexicons.INTENT_ARCHIVE_VERB, re.IGNORECASE)
# Bare-noun archive phrasings that need no explicit target ("get me the
# archive", "as a single file").
_ARCHIVE_NOUN_RE = re.compile(lexicons.INTENT_ARCHIVE_NOUN, re.IGNORECASE)
# Any archive-format mention (broad) — recognises an archive request even
# without an archive verb ("give me a tar of the project").
_ARCHIVE_FORMAT_RE = re.compile(lexicons.INTENT_ARCHIVE_FORMAT, re.IGNORECASE)
# The ONLY archive formats we can create → the two we offer. Any other format
# named (tar/rar/…) is treated as "format not yet chosen" so we ask.
_SUPPORTED_ARCHIVE_RE = re.compile(lexicons.INTENT_SUPPORTED_ARCHIVE,
                                   re.IGNORECASE)
# Generalized Claude-like clarifier rules (AechitectureLikeClaude.md):
# named doc genres, non-code deliverables, artifact references, vague asks.
_DOC_GENRE_RE = re.compile(lexicons.INTENT_DOC_GENRE, re.IGNORECASE)
_NONCODE_DELIVERABLE_RE = re.compile(lexicons.INTENT_NONCODE_DELIVERABLE,
                                     re.IGNORECASE)
_ARTIFACT_REF_RE = re.compile(lexicons.INTENT_ARTIFACT_REF, re.IGNORECASE)
_ANALYZE_VERB_RE = re.compile(lexicons.INTENT_ANALYZE_VERB, re.IGNORECASE)
_VAGUE_IMPERATIVE_RE = re.compile(lexicons.INTENT_VAGUE_IMPERATIVE,
                                  re.IGNORECASE)


def _canonical_archive_format(text: str) -> str | None:
    """Return 'zip' or '7z' if the text names a SUPPORTED archive format, else
    None (so an unsupported/omitted format triggers the clarification)."""
    m = _SUPPORTED_ARCHIVE_RE.search(text or "")
    if not m:
        return None
    return "7z" if "7" in m.group(0).lower() else "zip"


@dataclass
class Assessment:
    """Result of the deterministic pre-gate."""
    intent: str
    slots: dict
    missing_required: list[str]
    missing_optional: list[str]
    answerable: bool
    estimated_quality: float        # 0..1, answer quality without clarifying
    ambiguity: float                # 0..1
    confidence: float               # 0..1 composed (how sure we can answer now)
    clarification_gain: float       # 0..1
    clarification_cost: float       # 0..1
    decision: str                   # ANSWER | CLARIFY | DEFER
    suppressed: list[str] = field(default_factory=list)  # known slots: never ask
    reasons: list[str] = field(default_factory=list)     # uncertainty attribution
    strategy: str = "answer"        # answer | clarify | plan
    reason: str = ""                # short human-readable summary
    # Phase-1 decision core (additive; None/neutral when the flags are off).
    matrix: object | None = None    # clarify.requirement_matrix.RequirementMatrix
    risk: float = 0.0               # 0..1 numeric risk (clarify.risk)
    risk_level: str = "low"         # low | medium | high
    risk_band_delta: float = 0.0    # answer-band nudge downstream bands apply
    # Phase-3: the policy decision record (rules fired + scores) for tracing.
    policy: dict | None = None


def detect_intent_smart(text: str, *, embed_fn=None) -> str:
    """PRIMARY intent classifier: the embeddings-based SEMANTIC classifier
    (bge-m3) understands intent by meaning; the keyword regex is only the
    deterministic FALLBACK for low-confidence and model-unavailable cases.

    Decision (when `cfg.semantic_intent.enabled`):
      1. embed → nearest-exemplar (intent, cosine);
      2. cosine ≥ primary_threshold → the semantic intent is authoritative
         (keyword rules do NOT participate);
      3. below that → consult the regex net; use the semantic best-guess only if
         the regex has no opinion (UNKNOWN);
      4. embedder unavailable → pure regex.
    Feature off → pure regex. Fail-open throughout. `embed_fn` is injectable for
    tests.
    """
    try:
        from app.core.config_loader import cfg
        sc = cfg.semantic_intent
        if not getattr(sc, "enabled", True):
            return detect_intent(text)                 # feature off → regex net
        from app.clarify import intent_semantic
        sem = intent_semantic.classify(text, embed_fn=embed_fn)
        if sem is None:                                # model unavailable → regex net
            return detect_intent(text)
        intent, score = sem
        _thr = float(getattr(sc, "primary_threshold", 0.50))
        # G1: the threshold adapts to outcomes when calibration is on (else the
        # configured literal). Fail-open to the literal.
        try:
            from app.core.calibration import calibrated
            _thr = calibrated("intent_threshold", _thr)
        except Exception:  # noqa: BLE001
            pass
        if score >= _thr:
            # Question-form correction: topic overlap with a coding exemplar
            # ("how does binary search work" ≈ "implement binary search…")
            # must never turn a definitional QUESTION into code generation —
            # that would trigger a "which language?" ask on a knowledge turn.
            if intent == INTENT_CODE_GEN \
                    and re.match(r"^\s*(what|why|when|where|who|how)\b",
                                 text or "", re.IGNORECASE) \
                    and not re.search(
                        r"\b(write|create|build|implement|generate|"
                        r"give me|make me|code up)\b", text or "",
                        re.IGNORECASE):
                return INTENT_KNOWLEDGE
            return intent                              # SEMANTIC decides (no keywords)
        # Low-confidence: let the deterministic net try; fall back to the
        # semantic best-guess only when the regex itself has no opinion.
        rx = detect_intent(text)
        return rx if rx != INTENT_UNKNOWN else intent
    except Exception:  # noqa: BLE001 — never break the gate on the semantic path
        return detect_intent(text)


def detect_intent(text: str) -> str:
    """Single best-fit intent for the latest request (deterministic)."""
    t = (text or "").strip()
    if not t:
        return INTENT_CHITCHAT
    if _GREETING_RE.match(t):
        return INTENT_CHITCHAT
    # Archive/compress an EXISTING deliverable ("compress this", "zip up the
    # project", "compressed file of the whole project"). Requires an archive
    # verb + a target referring to existing content, and must NOT be a code task
    # (a named language or a code-artifact noun like function/script means
    # "write code that zips", not "compress my project").
    # Not a code task (a named language or a code-artifact noun means "write
    # code that zips", not "compress my project"), and not a definitional
    # question ("what is a zip code" / "how does compression work") or the
    # "zip code" compound — those are knowledge, not an archive request.
    _definitional_q = re.match(
        r"^\s*(what|which|who)\b.{0,40}?\b(is|are|means?|does|do)\b",
        t, re.IGNORECASE) is not None
    _not_code_task = (
        _TECH_RE.search(t.lower()) is None
        and not _definitional_q
        and not re.search(r"\bzip\s*codes?\b", t, re.IGNORECASE)
        and not re.search(
            r"\b(function|method|program|script|class|snippet|library|"
            r"module|util(?:ity)?|algorithm|implement)\b", t, re.IGNORECASE))
    _has_target = re.search(
        r"\b(this|it|that|project|projects|file|files|folder|"
        r"everything|whole|code|codebase|repo|repository|all|source)\b",
        t, re.IGNORECASE) is not None
    if _not_code_task and (
            # (a) an archive verb/format + a target ("zip the project"), OR
            ((_ARCHIVE_VERB_RE.search(t) or _ARCHIVE_FORMAT_RE.search(t))
             and _has_target)
            # (b) a bare-noun archive phrasing needing no target ("get me the
            #     archive", "as a single file") — archive intent isn't tied to
            #     any single keyword like "download".
            or _ARCHIVE_NOUN_RE.search(t)):
        return INTENT_ARCHIVE
    # Read-only summarization of existing content ("summarize this document")
    # is KNOWLEDGE, never a docs-generation ask — the design doc's Default
    # Knowledge rule: "Summarize this PDF → just summarize", no format Q.
    if re.match(r"^\s*(summari[sz]e|tl;?dr)\b", t, re.IGNORECASE):
        return INTENT_KNOWLEDGE
    # Test generation as the DIRECT OBJECT ("generate unit tests for this
    # service") wins over the project rule — "generate"+"service" would
    # otherwise misread as a project build and ask for a stack.
    if re.match(r"^\s*(generate|write|create|add|make)\s+(?:me\s+)?"
                r"(?:some\s+)?(?:unit|integration|e2e|api)?\s*tests?\b",
                t, re.IGNORECASE):
        return INTENT_TEST_GEN
    # A document GENRE as the direct object ("generate a README for the
    # project", "write a changelog") is a DOCS request with the format already
    # implied — not a project build asking for a stack.
    _m_doc = re.match(r"^\s*(generate|write|create|make|draft|prepare)\s+"
                      r"(?:me\s+)?(?:a|an|the)?\s*", t, re.IGNORECASE)
    if _m_doc and _DOC_GENRE_RE.match(t[_m_doc.end():]):
        return INTENT_DOCS
    # Build BEFORE generic code so "build an app" beats "build"=code verb.
    if _PROJECT_VERB_RE.search(t) and _PROJECT_NOUN_RE.search(t):
        return INTENT_PROJECT_BUILD
    if _DEBUG_RE.search(t):
        return INTENT_DEBUGGING
    if _TEST_RE.search(t):
        return INTENT_TEST_GEN
    if _DOCS_RE.search(t):
        return INTENT_DOCS
    if _COMPARISON_RE.search(t):
        return INTENT_COMPARISON
    if _DESIGN_RE.search(t):
        return INTENT_DESIGN
    # Read-only EXPLAIN of existing/pasted content wins over code-generation:
    # "explain this program", "review this code", or a pasted snippet + an
    # explanation verb → answer directly, never ask which language to WRITE it in.
    if _EXPLAIN_EXISTING_RE.search(t) or (
            _looks_like_pasted_code(t) and _KNOWLEDGE_RE.search(t)):
        return INTENT_KNOWLEDGE
    # Question-form knowledge asks ("what is a hash map?", "how does sorting
    # work?") are definitional — never code-generation, even though they can
    # contain an operation word (hash/sort/search) that would otherwise match.
    if re.match(r"^\s*(what|why|when|where|who|how)\b", t, re.IGNORECASE) \
            and _KNOWLEDGE_RE.search(t):
        return INTENT_KNOWLEDGE
    # Code generation: a build/write verb or a concrete operation.
    if _CODE_GEN_RE.search(t) or _OPERATION_RE.search(t):
        return INTENT_CODE_GEN
    if _KNOWLEDGE_RE.search(t):
        return INTENT_KNOWLEDGE
    return INTENT_UNKNOWN


def extract_slots(text: str, recent: str = "") -> dict:
    """Pull the meaningful slots from the request + recent window."""
    blob = f"{recent} {text}".lower()
    tech = [m.group(0).lower() for m in _TECH_RE.finditer(blob)]
    frameworks = [m.group(0).lower() for m in _FRAMEWORK_RE.finditer(blob)]
    # A "language" is any tech token that is not purely a framework.
    fw_set = set(frameworks)
    language = next((x for x in tech if x not in fw_set), None)
    framework = frameworks[0] if frameworks else None
    platform_m = _PLATFORM_RE.search(text or "")
    operation_m = _OPERATION_RE.search(text or "")
    constraints = sorted({m.group(0).lower()
                          for m in _CONSTRAINT_RE.finditer(text or "")})
    fmt_m = _DOC_FORMAT_RE.search(text or "")
    return {
        "language": language,
        "framework": framework,
        "platform": platform_m.group(0).lower() if platform_m else None,
        "operation": operation_m.group(0).lower() if operation_m else None,
        "doc_format": fmt_m.group(0).lower() if fmt_m else None,
        # Only a SUPPORTED format (zip/7z) counts as "named"; tar/rar/etc. → None
        # so the clarification still fires and offers the two we can create.
        "archive_format": _canonical_archive_format(text or ""),
        "constraints": constraints,
        "has_tech": bool(tech),
    }


def _required_optional(intent: str, slots: dict) -> tuple[list[str], list[str]]:
    """Required-vs-optional missing fields for the intent (the layer most
    assistants get wrong). Only project builds have a hard required field."""
    missing_req: list[str] = []
    missing_opt: list[str] = []
    if intent == INTENT_PROJECT_BUILD:
        if not slots.get("has_tech"):
            missing_req.append("language_or_framework")
        if not slots.get("platform"):
            missing_opt.append("platform")
    elif intent == INTENT_CODE_GEN:
        # A code deliverable's language materially changes everything, so when
        # the user names NEITHER a language NOR a framework (which implies one),
        # it's a REQUIRED missing choice → ask which language rather than
        # silently defaulting to Python.
        if not slots.get("language") and not slots.get("framework"):
            missing_req.append("language")
    elif intent == INTENT_DOCS:
        # "Document this" with no stated format/scope is genuinely ambiguous
        # (README vs PDF vs Word vs inline comments) → ask before generating.
        if not slots.get("doc_format"):
            missing_req.append("doc_format")
    elif intent == INTENT_ARCHIVE:
        # "Compress this / give me the archive" with no format named → ask which
        # archive format (zip / tar.gz / 7z) before producing it.
        if not slots.get("archive_format"):
            missing_req.append("archive_format")
    # knowledge / comparison / design / debugging / test / chitchat have no hard
    # requirement — a useful answer (possibly with stated assumptions) is always
    # producible.
    return missing_req, missing_opt


def _ambiguity(text: str, intent: str, slots: dict) -> float:
    """0..1 ambiguity: vague wording, ultra-short prompts, open-ended builds."""
    t = (text or "").strip()
    score = 0.0
    if _VAGUE_RE.search(t):
        score += 0.4
    words = len(t.split())
    if words <= 2 and intent not in (INTENT_CHITCHAT,):
        score += 0.3
    if intent == INTENT_PROJECT_BUILD and not slots.get("has_tech"):
        score += 0.4
    if intent == INTENT_UNKNOWN:
        score += 0.4
    return min(1.0, score)


def _answerability(intent: str, missing_req: list[str],
                   ambiguity: float) -> tuple[bool, float]:
    """Can a useful answer be produced now, and at what estimated quality?"""
    if intent == INTENT_UNKNOWN:
        return (ambiguity < 0.5, max(0.0, 0.6 - ambiguity))
    answerable = len(missing_req) == 0
    quality = 1.0 - 0.3 * len(missing_req) - 0.25 * ambiguity
    return answerable, max(0.0, min(1.0, quality))


def _slot_score(intent: str, slots: dict) -> float:
    """Fraction of the slots that matter for this intent that are filled."""
    if intent == INTENT_PROJECT_BUILD:
        have = sum(bool(slots.get(k)) for k in ("language", "framework",
                                                "platform"))
        return min(1.0, have / 2.0)        # language/framework + platform
    if intent == INTENT_CODE_GEN:
        have = sum(bool(slots.get(k)) for k in ("language", "operation",
                                                "constraints"))
        return min(1.0, 0.5 + 0.25 * have)  # answerable baseline + specificity
    return 1.0  # non-code intents don't depend on these slots


def _compose_confidence(intent: str, slots: dict, missing_req: list[str],
                        answerable: bool, ambiguity: float,
                        has_context: bool) -> float:
    """Weighted composition from the analysis doc:
        intent*.25 + slots*.20 + completeness*.20 + answerability*.20
        + (1-ambiguity)*.10 + context*.05
    """
    intent_score = 0.4 if intent == INTENT_UNKNOWN else 1.0
    slot_score = _slot_score(intent, slots)
    completeness = 1.0 - min(1.0, 0.5 * len(missing_req))
    answer_score = 1.0 if answerable else 0.3
    ambiguity_score = 1.0 - ambiguity
    context_score = 1.0 if has_context else 0.7
    c = (intent_score * 0.25 + slot_score * 0.20 + completeness * 0.20
         + answer_score * 0.20 + ambiguity_score * 0.10 + context_score * 0.05)
    return max(0.0, min(1.0, c))


def _gain(missing_req: list[str], missing_opt: list[str]) -> float:
    """Information gain of asking now: high when a REQUIRED field is missing."""
    return min(1.0, 0.6 * len(missing_req) + 0.15 * len(missing_opt))


def _cost(intent: str, answerable: bool) -> float:
    """Friction cost of interrupting. Higher when we could already answer."""
    return 0.6 if answerable else 0.3


def assess(text: str, recent: str = "",
           known_prefs: dict | None = None, *,
           has_artifact: bool = False,
           attachment_slots: dict | None = None) -> Assessment:
    """Run the full deterministic pre-gate and return an [Assessment].

    Decision policy (the doc's final gate):
        if answerable and no required field missing and gain < cost → ANSWER
        elif a required field is missing                            → CLARIFY
        else                                                        → DEFER

    `has_artifact` — the caller knows this turn carries an attachment /
    uploaded file / image, so "analyze my code"-style asks are answerable.
    `attachment_slots` — slots DETECTED INSIDE the uploaded content (a
    StackProfile: language/framework from manifests) — Phase-2 clarification
    elimination: an uploaded Spring project means "add auth" never asks which
    language. They satisfy required slots exactly like known preferences, and
    the requirement matrix attributes them to source="attachment".
    """
    known = {k: v for k, v in (known_prefs or {}).items() if v}
    _att = {k: v for k, v in (attachment_slots or {}).items() if v}
    known.update(_att)          # detected-in-upload evidence suppresses asks
    # `recent` must be a string (downstream does `recent.strip()` / regex on
    # it). Coerce defensively — a caller that passes a list of recent messages
    # would otherwise raise AttributeError, and callers wrap assess() in a
    # broad try/except, so the failure would be silent (this exact class of
    # bug hid the empty-target detection for months).
    if not isinstance(recent, str):
        try:
            recent = " ".join(str(x) for x in recent) if recent else ""
        except Exception:  # noqa: BLE001
            recent = ""
    # Tiered: regex + semantic (when enabled). Fail-open to pure regex.
    intent = detect_intent_smart(text)
    slots = extract_slots(text, recent)
    # Correct semantic ARCHIVE misfires. The embedding classifier maps things
    # like "get me a PDF document" or "Migrate Spring MVC to Spring Boot" to
    # ARCHIVE, which then wrongly demands an archive format. ARCHIVE requires
    # deterministic evidence (an archive verb or format actually present):
    if intent == INTENT_ARCHIVE and not slots.get("archive_format") \
            and not _ARCHIVE_VERB_RE.search(text or "") \
            and not _ARCHIVE_FORMAT_RE.search(text or "") \
            and not _ARCHIVE_NOUN_RE.search(text or ""):
        # A named DOCUMENT format means it's a document request; anything else
        # falls back to the regex intent (never a spurious archive ask).
        intent = INTENT_DOCS if slots.get("doc_format") else detect_intent(text)
    # (Question-form CODE_GEN correction now lives inside detect_intent_smart
    #  so every consumer of the classifier benefits, not just this gate.)
    # Self-critique / suppression: anything already named (this turn, recent, or
    # durable prefs) must NEVER be asked again.
    suppressed: list[str] = []
    if slots.get("language"):
        suppressed.append("language")
    if slots.get("framework"):
        suppressed.append("framework")
    if slots.get("platform"):
        suppressed.append("platform")
    for k in known:
        if k not in suppressed:
            suppressed.append(k)

    missing_req, missing_opt = _required_optional(intent, slots)
    # A known preference can satisfy a build's required tech slot.
    if "language_or_framework" in missing_req and (
            known.get("language") or known.get("framework")
            or known.get("stack")):
        missing_req.remove("language_or_framework")
    # A known language/framework preference also satisfies a code-gen request.
    if "language" in missing_req and (
            known.get("language") or known.get("framework")
            or known.get("stack")):
        missing_req.remove("language")

    # ---- Generalized Claude-like rules (AechitectureLikeClaude.md) --------
    # The decision-matrix rules the 150+ scenario families reduce to:
    # missing artifact, missing subject, named genre = satisfied choice, and
    # "vague imperative" = under-specified task. All deterministic + tested
    # against the scenario catalog (tests/data/clarifier_scenarios.json).
    t_low = (text or "").lower()
    words = len((text or "").split())
    artifact_present = (
        has_artifact
        or _looks_like_pasted_code(text)
        or "[attached" in t_low          # upload path injects this marker
        or "```" in (recent or "")       # code earlier in the conversation
        # "Here is the stack trace / code below" — they're providing it.
        or bool(re.search(r"\b(here(?:'s| is| are)|below|as follows|"
                          r"attached|pasted|following)\b", t_low))
        # Inline error evidence: the prompt ITSELF contains the exception /
        # traceback / location ("what does this error mean:
        # NullPointerException at line 42") — the referenced artifact is
        # right there, never ask for it. CamelCase match is deliberate so a
        # bare "fix my error" doesn't count as evidence.
        or bool(re.search(r"[A-Z]\w*(?:Exception|Error)\b|\bTraceback\b|"
                          r"\bat line \d+", text or ""))
        or "stack trace" in t_low or "stacktrace" in t_low
    )
    _genre = bool(_DOC_GENRE_RE.search(t_low))
    _deliverable = bool(_NONCODE_DELIVERABLE_RE.search(t_low))
    # Named document GENRE ("API design document", "PRD", "test plan") — the
    # user already said what to produce; format has a safe default. Never ask.
    if "doc_format" in missing_req and _genre:
        missing_req.remove("doc_format")
    # Non-code deliverable ("acceptance criteria", "user stories", "UML"):
    # never a "which language?" ask. A bare deliverable with no subject at all
    # ("Generate requirements.") is missing its SUBJECT instead.
    if _deliverable:
        for _k in ("language", "language_or_framework"):
            if _k in missing_req:
                missing_req.remove(_k)
        if words <= 3 and not artifact_present:
            missing_req.append("subject")
    # Working ON provided code ("apply a factory pattern to this code" with the
    # code attached/pasted): the language is whatever the artifact is written
    # in — asking "which language?" would be an unnecessary clarification.
    if ("language" in missing_req and artifact_present
            and _ARTIFACT_REF_RE.search(text or "")):
        missing_req.remove("language")
    # Analyze-type ask about the user's OWN artifact with nothing attached or
    # pasted ("Fix my code", "Analyze the screenshot") → the artifact itself is
    # the missing required input; ask for it instead of hallucinating one.
    if (not artifact_present
            and _ARTIFACT_REF_RE.search(text or "")
            and (_ANALYZE_VERB_RE.search(t_low)
                 or intent in (INTENT_DEBUGGING, INTENT_TEST_GEN))):
        missing_req.append("artifact")
    # Test generation with no code in sight, no tooling named, and no described
    # subject ("Write unit tests.") — you can't test code you don't have.
    if (intent == INTENT_TEST_GEN
            and "artifact" not in missing_req
            and not artifact_present
            and not slots.get("has_tech")
            and not re.search(r"\bfor\b\s+\w", t_low)):
        missing_req.append("artifact")
    # A bare incident report ("Production is down.") — the evidence (logs,
    # error, symptoms) is the missing required input.
    if (intent == INTENT_DEBUGGING
            and "artifact" not in missing_req
            and words <= 4
            and not artifact_present
            and not slots.get("has_tech")):
        missing_req.append("artifact")
    # Vague imperative: a very short command naming no tech, no concrete
    # operation, no artifact and no format ("Deploy my application.", "Design a
    # database.", "Add monitoring.") is under-specified — one targeted question
    # beats guessing a stack/scope. Knowledge questions ("what is docker?") are
    # NOT vague: explanation verbs are absent from the imperative lexicon. A
    # named deliverable/genre isn't vague either. The leading imperative verb
    # itself doesn't count as a concrete operation ("Write a query" is vague;
    # "reverse a string" is concrete).
    _op = slots.get("operation")
    _op_is_leading = bool(
        _op and t_low.strip().startswith(_op))
    # A definitional KNOWLEDGE question ("what is a hashmap?") is answerable,
    # not vague — but only in actual QUESTION FORM (interrogative start or a
    # trailing "?"); a semantic mislabel on an imperative ("Migrate my
    # application" → knowledge) must not slip past the vague rule. And
    # recommendation-style asks ("what should I use for caching?") stay vague.
    _definitional = (
        intent == INTENT_KNOWLEDGE
        and bool(re.match(r"^\s*(what|why|when|where|who|how)\b", t_low)
                 or t_low.rstrip().endswith("?"))
        and not re.search(r"\b(should|recommend|best|choose|pick|use)\b",
                          t_low))
    if (not missing_req
            and words <= 5
            and not _definitional
            and _VAGUE_IMPERATIVE_RE.search(text or "")
            and not slots.get("has_tech")
            # A DESIGN ask with a genuinely COMPOUND subject ("design a
            # payment system architecture", 5+ words) is answerable with
            # stated assumptions — the doc's "ask scale only if necessary".
            # Shorter design asks ("Design a chat application.") stay
            # under-specified per the scenario catalog: one targeted scope
            # question beats guessing.
            and not (intent == INTENT_DESIGN and words >= 5)
            # A stack the user already decided (this conversation or durable
            # preference) makes a short build command actionable, not vague.
            and not (known.get("language") or known.get("framework")
                     or known.get("stack"))
            and not (_op and not _op_is_leading)
            and not slots.get("doc_format")
            and not _genre
            and not _deliverable
            and not artifact_present
            and not _GREETING_RE.search(text or "")):
        missing_req.append("task_details")

    ambiguity = _ambiguity(text, intent, slots)
    answerable, quality = _answerability(intent, missing_req, ambiguity)
    has_context = bool((recent or "").strip()) or bool(known)
    confidence = _compose_confidence(intent, slots, missing_req, answerable,
                                     ambiguity, has_context)
    gain = _gain(missing_req, missing_opt)
    cost = _cost(intent, answerable)

    reasons: list[str] = []
    for f in missing_req:
        reasons.append(f"missing_required:{f}")
    if ambiguity >= 0.5:
        reasons.append("high_ambiguity")

    # ---- Final gate -------------------------------------------------------
    # UNKNOWN intent never answer-firsts deterministically: we couldn't even
    # classify what's being asked, so the LLM gate decides (DEFER) — it can
    # still answer, but a confident deterministic ANSWER would be unearned.
    #
    # Phase-3: the gate is now POLICY-DRIVEN (app/policy) — builtin rules
    # replicate the legacy cascade below EXACTLY, and config `policy.rules`
    # can add/override declaratively. The decision record (which rules fired,
    # scores) lands on Assessment.policy for the trace. Fail-open / flag-off →
    # the legacy cascade runs verbatim.
    _policy_record: dict | None = None
    decision = strategy = reason = None
    try:
        from app.core.config_loader import cfg as _pcfg
        if getattr(_pcfg.policy, "enabled", True):
            from app.policy import (ACTION_ANSWER, ACTION_CLARIFY, decide)
            _pd = decide({
                "intent": intent,
                "answerable": answerable,
                "missing_required": missing_req,
                "missing_optional": missing_opt,
                "ambiguity": ambiguity,
                "confidence": confidence,
                "clarification_gain": gain,
                "clarification_cost": cost,
                "estimated_quality": quality,
                "has_artifact": has_artifact,
                "has_context": has_context,
                "slots": slots,
            })
            _policy_record = _pd.as_dict()
            if _pd.action == ACTION_ANSWER:
                decision, strategy = ANSWER, "answer"
                reason = _pd.reason or "Request is specific enough to answer directly."
            elif _pd.action == ACTION_CLARIFY:
                decision = CLARIFY
                strategy = "plan" if intent == INTENT_PROJECT_BUILD else "clarify"
                reason = (("A required detail is missing: "
                           + ", ".join(missing_req)) if missing_req
                          else (_pd.reason or "Clarification policy fired."))
            else:
                decision, strategy = DEFER, "answer"
                reason = _pd.reason or "Likely answerable; defer wording to the gate."
    except Exception:  # noqa: BLE001 — policy engine must never break the gate
        decision = None
    if decision is None:      # legacy cascade (flag off / engine failure)
        if (answerable and not missing_req and gain < cost
                and intent != INTENT_UNKNOWN):
            decision, strategy = ANSWER, "answer"
            reason = "Request is specific enough to answer directly."
        elif missing_req:
            decision = CLARIFY
            # Large open-ended builds are better served by a plan-first ask.
            strategy = "plan" if intent == INTENT_PROJECT_BUILD else "clarify"
            reason = "A required detail is missing: " + ", ".join(missing_req)
        else:
            decision, strategy = DEFER, "answer"
            reason = "Likely answerable; defer wording to the gate."

    out = Assessment(
        intent=intent,
        slots=slots,
        missing_required=missing_req,
        missing_optional=missing_opt,
        answerable=answerable,
        estimated_quality=quality,
        ambiguity=ambiguity,
        confidence=confidence,
        clarification_gain=gain,
        clarification_cost=cost,
        decision=decision,
        suppressed=suppressed,
        reasons=reasons,
        strategy=strategy,
        reason=reason,
        policy=_policy_record,
    )
    # ---- Phase-1 decision core (additive, flag-gated, fail-open) -----------
    # Neither enrichment changes the decision computed above: the matrix
    # mirrors the flat lists 1:1 (provenance for policies/traces/Phase-2
    # slot-filling), and risk only exports a band DELTA that the downstream
    # confidence bands may apply.
    try:
        from app.core.config_loader import cfg as _cfg
        if getattr(_cfg.decision_core, "requirement_matrix", True):
            from app.clarify.requirement_matrix import (SOURCE_ATTACHMENT,
                                                        build_matrix)
            out.matrix = build_matrix(
                intent, slots, missing_req, missing_opt,
                text_slots=extract_slots(text or "", ""),
                known_prefs=known)
            # Attachment-detected slots carry their true provenance (they were
            # merged into `known` above only for suppression).
            for _k, _v in _att.items():
                out.matrix.fill(_k, str(_v), SOURCE_ATTACHMENT)
        if getattr(_cfg.decision_core, "risk_scoring", True):
            from app.clarify.risk import assess_risk
            ra = assess_risk(text or "", intent, slots)
            out.risk, out.risk_level = ra.score, ra.level
            out.risk_band_delta = ra.band_delta
            if ra.level == "high":
                out.reasons.append("high_risk:" + ",".join(
                    r for r in ra.reasons if not r.startswith("intent_base")))
    except Exception:  # noqa: BLE001 — enrichment must never break the gate
        pass
    # Phase-6 decision metrics (central counters; fail-open, near-zero cost).
    try:
        from app.obs.decision_metrics import record_gate_decision
        record_gate_decision(out.decision,
                             (out.policy or {}).get("rule_id"))
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = [
    "Assessment", "assess", "detect_intent", "detect_intent_smart", "extract_slots",
    "ANSWER", "CLARIFY", "DEFER",
    "INTENT_CHITCHAT", "INTENT_KNOWLEDGE", "INTENT_COMPARISON",
    "INTENT_DEBUGGING", "INTENT_TEST_GEN", "INTENT_DOCS", "INTENT_DESIGN",
    "INTENT_CODE_GEN", "INTENT_PROJECT_BUILD", "INTENT_UNKNOWN",
    "INTENT_ARCHIVE",
]
