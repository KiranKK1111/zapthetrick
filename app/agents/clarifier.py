"""Clarifier — decides if a request is ambiguous and, if so, produces a
Claude-style clarification payload.

Runs on every substantive message — it is the sole decision-maker (no upstream
keyword gate). It makes one fast JSON LLM call that *declines* on most messages,
including greetings / acknowledgements / trivial turns, and only asks when a
clarifying question genuinely helps. When it does ask, the Supervisor forwards a
structured `clarify` event to the UI and the client renders interactive option
cards ([widgets/clarification_panel.dart]).

Output contract — TWO blackboard slots:

  • `clarifying_questions` (list[dict]) — the questions (back-compat shape):
        {
          "id": "q1",
          "question": "Which language should the program be in?",
          "header": "Language",
          "kind": "single" | "multi" | "rank",
          "multiSelect": false,            # retained for older clients
          "reason": "why this matters",    # optional per-question rationale
          "options": [
            {"id": "o1", "label": "Python", "description": "...",
             "recommended": true},
            ...
          ]
        }
    An empty list means "no clarification needed — answer directly".

  • `clarify_meta` (dict) — turn-level metadata (all keys defaulted):
        {
          "confidence": 0.0..1.0,           # how sure we can answer w/o asking
          "blocking": bool,                 # withhold the answer until answered?
          "reason": str,                    # top-level "why I'm asking"
          "estimated_questions_saved": int,
          "mode": "ask" | "assume" | "sample",
          "assumptions": [{"id","label","value"}],  # only when mode == "assume"
          "sample": bool
        }

Everything beyond `question`/`options` is additive and optional so older clients
and the existing parse tests keep working.
"""
from __future__ import annotations

import contextlib
import json
import re

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_INTENT, KEY_QUESTION
from ..blackboard.scheduler import P0
from ..chat.difficulty import STANDARD
from ..clarify.intent_pipeline import ANSWER, CLARIFY, INTENT_ARCHIVE, assess
from ..clarify.calibration import calibrate
from ..clarify.adaptation import adapted_answer_band
from ..clarify.interpretations import parse_interpretations, pick_interpretation
from ..clarify.simulation import questions_to_assumptions
from ..clarify.critic import review as _critic_review
from ..clarify.latent import suggest as _latent_suggest
from ..core.llm_client import LLMError, llm
from .base import Agent

_MAX_QUESTIONS = 3
_MAX_OPTIONS = 4

# Slot the ClarifierAgent writes alongside `clarifying_questions`.
KEY_CLARIFY_META = "clarify_meta"

# Confidence-band thresholds (Requirement 2). Now sourced from central config
# (`cfg.confidence`) so they're tuned in one place; defaults equal the former
# literals (0.90 / 0.70 / 0.40), so behavior is unchanged until tuned.
def _band_high() -> float:
    from app.core.config_loader import cfg
    return cfg.confidence.band_high


def _band_assume() -> float:
    from app.core.config_loader import cfg
    return cfg.confidence.band_assume


def _band_targeted() -> float:
    from app.core.config_loader import cfg
    return cfg.confidence.band_targeted


def _planning_temp() -> float:
    from app.core.config_loader import temperature_for
    return temperature_for("planning")

_SYSTEM = (
    "You are a clarification gate for an AI assistant. Read the conversation so "
    "far and the user's LATEST request, then decide whether answering it WELL "
    "truly requires a specific missing detail you cannot reasonably infer or "
    "assume.\n"
    "\n"
    "Ask ONLY when a specific missing choice would MATERIALLY change the answer "
    "and you cannot infer it from the conversation. Otherwise return an empty "
    "list. DO NOT ask when:\n"
    "- the request is already specific enough to answer well;\n"
    "- the user already NAMED the choice in their request (e.g. 'the dart "
    "implementation', 'in Python', 'using React', 'a CLI tool') — that detail "
    "is GIVEN, never ask about it;\n"
    "- the missing detail is implied by earlier turns (follow-ups like 'make it "
    "shorter', 'now in Java', 'explain more');\n"
    "- it's a greeting, thanks, or chit-chat;\n"
    "- any reasonable default would satisfy the user.\n"
    "Most messages need NO clarification. Bias strongly toward answering.\n"
    "\n"
    "DECIDE LIKE THIS (answer-first gate): before asking anything, internally "
    "extract the request's slots — task/operation, programming language, "
    "framework/runtime, platform, and any constraints/techniques — then judge "
    "whether you could already produce an answer that satisfies ~85%+ of the "
    "user's intent. If YES, return clarify=false with high confidence and DO NOT "
    "ask. Only clarify when a REQUIRED slot is genuinely missing AND no "
    "reasonable default exists AND the missing choice would materially change "
    "the answer. A clarification you ask must have real information gain — never "
    "ask about something the user already stated or that you can safely assume.\n"
    "\n"
    "HARD SUPPRESSION (never violate): if the request is a self-contained coding "
    "task whose operation is clear (e.g. reverse a string, sort a list, parse "
    "JSON, write a regex, implement an algorithm, fix this snippet) AND a "
    "language is named or obvious from context, you MUST answer — clarify=false, "
    "confidence >= 0.95. Likewise never ask which language/framework/platform "
    "when the user already named it (explicitly or in an earlier turn).\n"
    "\n"
    "IMPORTANT EXCEPTION — build/create requests: when the user asks you to "
    "BUILD, CREATE, GENERATE, or SCAFFOLD software (an app, project, system, "
    "website, API, service, script, tool, or library) and they have NOT stated "
    "the key technical choices, you SHOULD ask — these choices materially change "
    "the whole deliverable and rarely have a single safe default. Ask about the "
    "ones that are genuinely unspecified and relevant, e.g.: programming "
    "language; framework / runtime; UI vs CLI vs web vs mobile platform; data "
    "storage / database; and any major library. Skip any choice the user already "
    "made or that an earlier turn clearly implies, and don't ask if they only "
    "want an explanation, design, or pseudo-code rather than real code.\n"
    "\n"
    "SCENARIO PLAYBOOK — when a clarifying question is genuinely warranted, pick "
    "the angle that best fits the request:\n"
    "- Multiple interpretations: ask which meaning when 2+ readings change the "
    "answer; include only plausible options.\n"
    "- Trade-offs / priorities: when goals compete (speed / cost / quality / "
    "scalability), ask the user to rank or choose (kind=rank).\n"
    "- Scope: when features or boundaries are open-ended, offer a multi-select "
    "of candidate items (kind=multi).\n"
    "- Expertise / format / depth: when the right depth depends on the audience, "
    "ask experience level, preferred format, or length.\n"
    "- Intent / goal: when the underlying goal is unclear, ask WHY (the "
    "objective), not just what. Decompose analogies ('Netflix-like', 'like X') "
    "into the specific aspect intended.\n"
    "- Assumptions: at medium confidence (0.70-0.90) prefer mode=\"assume\" and "
    "list inferred assumptions to confirm instead of open questions.\n"
    "- Risk / reversibility / ownership: for destructive or hard-to-reverse "
    "actions, ask for confirmation and offer a safer alternative; when the user "
    "may prefer to delegate, offer an option that lets you decide.\n"
    "- Effort level: for build requests with no stated maturity, ask "
    "Prototype / MVP / Production / Enterprise.\n"
    "- Success criteria: for 'improve / optimize / scale', ask how success is "
    "measured.\n"
    "- Mood / urgency: if the user signals distress and the kind of help is "
    "unclear, ask what help is most useful (fast answer / diagnosis / "
    "step-by-step / options) — never request personal disclosure.\n"
    "- Time horizon: when the recommendation hinges on timeframe, ask the "
    "optimization horizon.\n"
    "- Unknown-unknowns: for broad build/design tasks you MAY add ONE optional, "
    "non-blocking multi-select surfacing hidden needs (compliance, offline, "
    "multi-tenancy, accessibility, internationalization) with 'None of these' "
    "and 'Not sure'.\n"
    "- Sequencing: for a broad multi-part goal, ask which deliverable to start "
    "with (or rank them by priority).\n"
    "Stay within 1-3 questions total; never re-ask a choice already decided.\n"
    "\n"
    "When you DO ask, every question and option must be SPECIFIC and RELEVANT to "
    "THIS exact request and conversation — never generic boilerplate:\n"
    "- Ground each question in what the user actually asked; ask only about the "
    "detail(s) that genuinely block a good answer, in priority order.\n"
    "- Make options concrete, realistic, and appropriate to the user's evident "
    "context and skill level — don't offer choices that don't fit what they're "
    "doing, and don't pad to hit a count.\n"
    "- ADVISE like an expert: for each question, decide the BEST-FIT option for "
    "the user's described scenario and set \"recommended\": true on exactly that "
    "one option (at most one per question). In that option's description, say in "
    "a few words WHY it's the best choice here. The other options should note "
    "when they'd be a better fit. This turns the question into real guidance, "
    "not just a poll.\n"
    "- 1-3 questions max; each has 2-5 options; each option a brief (<= 10 word) "
    "description. 'header' is a 1-2 word chip label naming the decision. Give "
    "each question a short 'reason' explaining why the answer matters.\n"
    "\n"
    "Choose 'kind' by the SEMANTICS of the choice:\n"
    "- \"multi\": the user could reasonably want MORE THAN ONE (e.g. which "
    "language(s), features, platforms, formats, libraries). Prefer this whenever "
    "picking several is sensible.\n"
    "- \"single\": genuinely mutually-exclusive — picking two makes no sense "
    "(yes/no, exactly one runtime/region/version).\n"
    "- \"rank\": the user should ORDER the options by priority (e.g. rank "
    "cost/speed/quality). Use only when ordering is the answer.\n"
    "\n"
    "LANGUAGE: write every question, header, option label, option description, "
    "reason, and assumption in the SAME language as the user's latest request. "
    "If the language is unclear, use English.\n"
    "\n"
    "CONFIDENCE & BANDING: set 'confidence' (0.0-1.0) = how sure you can answer "
    "WELL without asking. If you ask any question, confidence MUST be below 0.90. "
    "When confidence is 0.70-0.90, prefer mode=\"assume\": instead of questions, "
    "list your inferred 'assumptions' (each {label, value}) for the user to "
    "confirm. Set 'blocking' true ONLY when a correct answer is impossible "
    "without the missing detail; otherwise blocking=false (the answer can stream "
    "while the user optionally refines). Set 'reason' to a brief overall "
    "explanation of why you're asking, and 'estimated_questions_saved' to how "
    "many back-and-forth turns the answers will save.\n"
    "\n"
    "AMBIGUITY — INTERPRETATION SET: if the request genuinely reads in 2+ "
    "materially different ways, ALSO include an 'interpretations' array of "
    "{reading, probability} (probabilities ~sum to 1). If one reading clearly "
    "dominates, still answer it; only when none dominates should you ask. Omit "
    "'interpretations' when the request has a single clear reading.\n"
    "\n"
    "Respond with STRICT JSON only, no prose, in exactly this shape:\n"
    '{"clarify": <bool>, "confidence": <number>, "blocking": <bool>, '
    '"reason": <str>, "estimated_questions_saved": <number>, '
    '"mode": "ask"|"assume", "assumptions": [{"label": <str>, "value": <str>}], '
    '"interpretations": [{"reading": <str>, "probability": <number>}], '
    '"questions": [{"question": <str>, "header": <str>, '
    '"kind": "single"|"multi"|"rank", "reason": <str>, "options": [{"label": '
    '<str>, "description": <str>, "recommended": <bool>}]}]}'
)

_VALID_KINDS = {"single", "multi", "rank"}
_VALID_MODES = {"ask", "assume", "sample"}

# Default metadata applied when the model omits / malforms a field (R3.4).
_META_DEFAULTS = {
    "confidence": 1.0,
    "blocking": False,
    "reason": "",
    "estimated_questions_saved": 0,
    "mode": "ask",
    "assumptions": [],
    "sample": False,
    "preview": False,
    # advanced-intent-reasoning additive fields (R5/R7).
    "suggestions": [],
    "interpretations": [],
}

# Recognise a request for a generic DEMO clarification popup (R21).
_SAMPLE_RE = re.compile(
    r"\b(sample|example|demo|dummy|mock)\b.{0,40}\b(popup|pop-up|clarif\w*|"
    r"question\s*card|question\s*popup)\b",
    re.IGNORECASE,
)

# Recognise a clearly destructive / hard-to-reverse operation (R15.1).
_RISKY_RE = re.compile(
    r"\b(delete|drop|truncate|wipe|erase|purge|destroy)\b[^.\n]{0,40}\b("
    r"all|every|everything|database|table|records?|data|users?|accounts?|"
    r"production|prod|bucket|index|collection|volume)\b"
    r"|\brm\s+-rf\b|\bdrop\s+(table|database|schema)\b"
    r"|\btruncate\s+table\b|\bformat\s+(the\s+|my\s+|a\s+)?(disk|drive)\b",
    re.IGNORECASE,
)
# Words that mean the user wants CODE/explanation about the op, not to run it —
# we then don't pop a destructive-action confirmation.
_CODEY_RE = re.compile(
    r"\b(how\s+(do|to|can)|write|generate|example|snippet|function|method|"
    r"explain|teach|what|why|difference|syntax|tutorial|sample\s+code)\b",
    re.IGNORECASE,
)


def _recent_history(extras: dict, limit: int = 6) -> str:
    """A compact transcript of the last few turns for context."""
    prior = (extras or {}).get("prior_messages") or []
    lines: list[str] = []
    for m in prior[-limit:]:
        role = (m.get("role") or "").strip() or "user"
        content = (m.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)


class ClarifierAgent(Agent):
    name = "clarifier"
    priority = P0
    expected_latency_ms = 1_200
    reads = frozenset({KEY_QUESTION, KEY_INTENT})
    writes = frozenset({"clarifying_questions", KEY_CLARIFY_META})

    async def run(self, board: Blackboard) -> None:
        question = board.get(KEY_QUESTION, "") or ""
        extras = board.get("extras", {}) or {}
        history = _recent_history(extras)
        # Set by the route when the turn is a build/project request that names
        # no language/framework — we then MUST ask (Claude-style), with a
        # deterministic fallback so a hesitant gate model can't skip it.
        build_priority = bool(extras.get("clarify_priority"))
        # Active clarification personality (Phase 5): explorer|builder|expert|
        # autopilot|teacher. Shapes how many / what kind of questions we ask.
        clarify_mode = (extras.get("clarify_mode") or "").strip().lower()

        # Deterministic fast-path: a request for a generic SAMPLE popup (R21)
        # returns dummy questions without an LLM call and never affects a task.
        if is_sample_popup_request(question):
            questions, meta = _sample_payload()
            board.write("clarifying_questions", questions, agent=self.name)
            board.write(KEY_CLARIFY_META, meta, agent=self.name)
            return

        # Deterministic guard: a clearly destructive / hard-to-reverse action
        # gets a confirm-or-safer-alternative card before proceeding (R15.1).
        if not build_priority and is_risky_operation_request(question):
            questions, meta = _risky_payload()
            board.write("clarifying_questions", questions, agent=self.name)
            board.write(KEY_CLARIFY_META, meta, agent=self.name)
            return

        # Answer-first pre-gate (deterministic intent/confidence pipeline).
        # When the request is already answerable with no REQUIRED slot missing
        # (e.g. "reverse a string in Java using streams" — language + technique
        # given), answer directly and skip the LLM gate entirely: faster, and it
        # eliminates the class of false clarifications. Build-priority turns and
        # genuinely under-specified requests fall through to the gate below.
        known_prefs = extras.get("clarify_prefs") or {}
        if not isinstance(known_prefs, dict):
            known_prefs = {}
        assessment = assess(
            question, history, known_prefs,
            # The turn carries uploaded files/images → "analyze my code" is
            # answerable; never ask the user to attach what they attached.
            has_artifact=bool(extras.get("has_attachments")),
            # Phase-2: slots detected INSIDE the upload (StackProfile) satisfy
            # required slots — never ask what the project already answers.
            attachment_slots=extras.get("attachment_slots") or None)
        # Phase-5 unified world state: project the assessment (goal, intent,
        # matrix, risk, policy record, capability view) onto the blackboard so
        # every mesh agent / the SSE layer / the trace read ONE picture of the
        # turn. Additive + fail-open.
        try:
            from app.core.world_state import KEY_TURN_STATE, TurnState
            board.write(KEY_TURN_STATE,
                        TurnState.from_assessment(
                            assessment, goal=question).as_dict(),
                        agent=self.name)
        except Exception:  # noqa: BLE001
            pass
        # Confidence calibration (R2): correct the raw pre-gate confidence using
        # the device user's observed outcomes (no-op until enough data exists).
        _cal_buckets = extras.get("clarify_calibration") or {}
        _cal_conf = calibrate(assessment.confidence, _cal_buckets) \
            if isinstance(_cal_buckets, dict) else assessment.confidence
        # Deterministic ask for a REQUIRED choice the user omitted (non-build):
        # a code request with no language, or a "document this" with no format.
        # Produced immediately (no LLM wait) and blocking, so we ASK FIRST
        # instead of assuming a default (e.g. Python) — and it reliably wins the
        # race against the first answer token. Build turns keep their own path.
        # "Operate on existing content" asks (archive/document-this) are
        # pointless when THIS chat has nothing to act on yet — asking "which
        # archive format?" on a first prompt is nonsensical. The route passes
        # has_prior_code / has_prior_content so we skip those asks and let the
        # model respond ("there's nothing to archive yet"). A self-contained
        # code request still asks for the language.
        _has_code = bool(extras.get("has_prior_code"))
        _has_content = bool(extras.get("has_prior_content"))
        # DECLINE OUTRIGHT on an "archive/export the project" turn when this
        # chat has no target at all — no prior code, no attachments, nothing
        # pasted. EVERY clarifying question is nonsensical here (the LLM gate
        # used to invent "Which project should be archived?" cards); the
        # answering model explains there's nothing to package yet instead.
        # This guards the source, so every entry path (agent mesh, upload
        # stream) is covered regardless of which extras the route passed.
        if (not build_priority and assessment.intent == INTENT_ARCHIVE
                and not _has_code and not bool(extras.get("has_attachments"))
                and "```" not in question):
            board.write("clarifying_questions", [], agent=self.name)
            board.write(KEY_CLARIFY_META, {
                **_META_DEFAULTS,
                "confidence": round(_cal_conf, 2),
                "reason": ("Nothing to archive or export in this chat yet — "
                           "answering with guidance instead of asking."),
            }, agent=self.name)
            return
        if not build_priority and assessment.decision == CLARIFY:
            _det: list[dict] | None = None
            _det_reason = ""
            # The "which language?" ask fires ONLY with genuine code-request
            # evidence — the intent classifier sometimes mislabels a statement
            # or a rendering follow-up ("in a tabular format") as code_gen, and
            # firing the language card on those is the exact over-ask the user
            # sees. A named language is already suppressed upstream.
            if ("language" in assessment.missing_required
                    and _has_code_request(question)):
                _det = language_choice_question()
                _det_reason = "Tell me the language and I'll write it."
            elif ("doc_format" in assessment.missing_required and _has_content
                  and _has_doc_signal(question)):
                _det = document_choice_question()
                _det_reason = "Tell me the document type and I'll generate it."
            elif "archive_format" in assessment.missing_required and _has_code:
                _det = archive_format_question()
                _det_reason = "Tell me the format and I'll compress it."
            if _det is not None:
                meta = {
                    **_META_DEFAULTS,
                    "confidence": round(min(_cal_conf, 0.4), 2),
                    "blocking": True,
                    "reason": _det_reason,
                    "estimated_questions_saved": 2,
                }
                board.write("clarifying_questions", _det, agent=self.name)
                board.write(KEY_CLARIFY_META, meta, agent=self.name)
                return
            # We DECLINED every deterministic card because the turn shows no
            # genuine code / document / archive request. If those were the ONLY
            # missing slots, the CLARIFY was a false positive from a misread
            # follow-up or statement — answer directly instead of falling
            # through to the LLM planner (which would re-invent a question).
            # Real content gaps (subject / artifact) are NOT in this set, so
            # they still fall through and get a proper dynamic question.
            _mreq = assessment.missing_required or []
            if _mreq and all(
                    k in ("language", "doc_format", "archive_format")
                    for k in _mreq):
                assessment.decision = ANSWER
                assessment.reason = (
                    "Answering directly — this is a follow-up / rendering "
                    "request, not a code or document deliverable.")
        # #12 answer-first gate v2: upgrade a borderline DEFER to answer-first
        # when the user's LEARNED answerability shows clarifying was usually
        # unnecessary. Never touches a required-slot CLARIFY. Gated + fail-open.
        if (not build_priority and assessment.decision not in (ANSWER, CLARIFY)):
            with contextlib.suppress(Exception):
                from ..clarify.answer_gate import (
                    enabled as _v2_on, should_upgrade_to_answer)
                if should_upgrade_to_answer(
                        assessment.decision, _cal_conf, _cal_buckets,
                        enabled=_v2_on()):
                    assessment.decision = ANSWER
                    assessment.reasons.append(
                        "answer-first v2: learned answerability")
        if not build_priority and assessment.decision == ANSWER:
            meta = {
                **_META_DEFAULTS,
                "confidence": round(_cal_conf, 2),
                "calibrated_confidence": round(_cal_conf, 2),
                "reason": assessment.reason,
                # R7: answer-first → offer likely follow-ups (non-blocking).
                "suggestions": _latent_suggest(assessment.intent,
                                               assessment.slots),
            }
            board.write("clarifying_questions", [], agent=self.name)
            board.write(KEY_CLARIFY_META, meta, agent=self.name)
            return

        questions: list[dict] = []
        meta: dict = dict(_META_DEFAULTS)
        try:
            user_block = ""
            if history:
                user_block += f"Conversation so far:\n{history}\n\n"
            # Known preferences (this conversation + durable cross-session) are
            # ALREADY DECIDED — tell the gate not to ask about them (R16/R17).
            known = extras.get("clarify_prefs") or {}
            if isinstance(known, dict) and known:
                decided = "; ".join(f"{k}={v}" for k, v in known.items() if v)
                if decided:
                    user_block += (
                        "Already decided (do NOT ask about these): "
                        f"{decided}\n\n"
                    )
            # Slots the deterministic pipeline already extracted from THIS
            # request — the gate must never ask about these (self-critique /
            # clarification-suppression, AnalysisOnIntentsAndConfidence).
            _detected = []
            for _k in ("language", "framework", "platform"):
                _v = assessment.slots.get(_k)
                if _v:
                    _detected.append(f"{_k}={_v}")
            if assessment.slots.get("operation"):
                _detected.append(f"operation={assessment.slots['operation']}")
            if _detected:
                user_block += (
                    "Already specified by the user (do NOT ask about these): "
                    f"{'; '.join(_detected)}\n\n"
                )
            user_block += f"Latest request: {question}\n\n"
            if clarify_mode:
                user_block += _MODE_DIRECTIVES.get(clarify_mode, "") + "\n\n"
            if build_priority:
                user_block += (
                    "IMPORTANT: This is a request to BUILD software and the "
                    "user has NOT specified the programming language or "
                    "framework. You MUST return clarify=true and ask concise, "
                    "PROJECT-SPECIFIC questions about the unspecified key "
                    "choices — at minimum the programming language and the "
                    "framework/runtime, plus any other genuinely-open choice "
                    "(e.g. UI type: web/desktop/mobile/CLI, or data storage). "
                    "Tailor the options to the described project.\n\n"
                )
            # Steer the gate when the deterministic pre-gate found a REQUIRED
            # input missing but has no canned card for it (a missing artifact,
            # an under-specified task, a deliverable with no subject). The gate
            # words ONE targeted ask for exactly that.
            elif assessment.decision == CLARIFY and assessment.missing_required:
                _hint = {
                    "artifact": (
                        "the user references their own code/screenshot/logs/"
                        "error but attached or pasted NOTHING — ask them to "
                        "paste or attach it (that is the only question needed)"
                    ),
                    "task_details": (
                        "the request is a very short, under-specified command "
                        "— ask the ONE most-unblocking question (e.g. which "
                        "stack/target/scope), with concrete options"
                    ),
                    "subject": (
                        "the request names a deliverable but not WHAT it is "
                        "for — ask for the subject/system it should cover"
                    ),
                }
                _keys = [k for k in assessment.missing_required if k in _hint]
                if _keys:
                    user_block += (
                        "IMPORTANT: The deterministic pre-gate decided a "
                        "REQUIRED input is missing. You MUST return "
                        "clarify=true with ONE targeted question. Missing: "
                        + "; ".join(f"{k} ({_hint[k]})" for k in _keys)
                        + ".\n\n"
                    )
            user_block += "Return the JSON now."
            raw = await llm.complete(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_block},
                ],
                # The gate runs on EVERY substantive turn and must DECIDE before
                # the answer's first token to interrupt in time — so route it to
                # a fast, capable model ("standard"), not a slow giant.
                options={"response_format_json": True,
                         "temperature": _planning_temp(),
                         "difficulty": STANDARD},
            )
            questions, meta = _parse_payload(raw)
        except (LLMError, Exception):  # noqa: BLE001
            # Any failure → ask nothing; the turn answers normally.
            questions, meta = [], dict(_META_DEFAULTS)

        # Calibrate the gate's confidence against observed outcomes (R2) before
        # banding, so banding decisions use the corrected value.
        if isinstance(_cal_buckets, dict) and _cal_buckets:
            _c = calibrate(meta.get("confidence", 1.0), _cal_buckets)
            meta["confidence"] = _c
            meta["calibrated_confidence"] = round(_c, 2)

        # Confidence-band clamping (R2): cap counts / drop on high confidence.
        # Fatigue + trust adaptation (R3/R4): lower the answer band as the user
        # has recently dealt with / skipped clarifications, so the system asks
        # less. Safety cards never reach here, so they are never suppressed.
        _fat = extras.get("clarify_fatigue") or {}
        # Per-conversation-state answer band (R6.3) is the base; fatigue/trust
        # (R3/R4) lowers it further. Falls back to the default high band.
        try:
            _base_band = float(extras.get("clarify_answer_band", _band_high()))
        except (TypeError, ValueError):
            _base_band = _band_high()
        _high_band = adapted_answer_band(
            _base_band,
            int(_fat.get("recent", 0) or 0),
            int(_fat.get("skips", 0) or 0),
            int(_fat.get("answers", 0) or 0),
        ) if isinstance(_fat, dict) else _base_band
        # Phase-1 risk nudge (clarify/risk.py): HIGH-risk work raises the bar
        # to answer without asking (+delta), cheap read-only work lowers it
        # (-delta/2). Small, centrally-capped, clamped to sane band range —
        # risk nudges the band machinery, it never overrides it. Neutral
        # (delta 0.0) when risk scoring is off or assessment failed.
        try:
            _rd = float(getattr(assessment, "risk_band_delta", 0.0) or 0.0)
            if _rd:
                _high_band = max(cfg.confidence.band_floor,
                                 min(0.98, _high_band + _rd))
                meta["risk"] = {
                    "score": round(float(getattr(assessment, "risk", 0.0)), 3),
                    "level": getattr(assessment, "risk_level", "low"),
                    "band_delta": round(_rd, 3),
                }
        except Exception:  # noqa: BLE001 — risk is an additive nudge only
            pass
        questions = _apply_band(questions, meta, _high_band)
        # Adaptive-mode clamping (R18): apply the active personality on top.
        questions = _apply_mode(questions, meta, clarify_mode)

        # ---- Phase-4 reasoning chain (advanced-intent-reasoning) ----------
        if not build_priority:
            # R5: if the gate returned an interpretation set, let a dominant
            # reading answer directly, or turn the readings into ONE precise
            # disambiguation question when none dominates.
            _kind, _payload = pick_interpretation(meta.get("interpretations")
                                                  or [])
            if _kind == "answer":
                questions = []
            elif _kind == "ask" and not questions and isinstance(_payload, list):
                questions = [{
                    "id": "disambiguate",
                    "question": "Which did you mean?",
                    "header": "Meaning",
                    "kind": "single",
                    "multiSelect": False,
                    "reason": "Your request could be read more than one way.",
                    "options": _payload,
                }]

            # R9: critic drops questions about already-known slots or with no
            # real information gain; if it empties the set, we answer.
            questions = _critic_review(questions, assessment.suppressed,
                                       known_prefs)

            # R8: at borderline (assumption-band) confidence, convert remaining
            # questions into stated assumptions the user can correct instead of
            # blocking on them.
            try:
                _conf = float(meta.get("confidence", 1.0))
            except (TypeError, ValueError):
                _conf = 1.0
            if questions and _conf >= _band_assume() and meta.get("mode") != "sample":
                _assumptions = questions_to_assumptions(questions)
                if _assumptions:
                    meta["mode"] = "assume"
                    meta["assumptions"] = _assumptions
                    questions = []

            # R7: when we end up answering (no questions), offer likely
            # follow-ups (non-blocking).
            if not questions and not meta.get("suggestions"):
                meta["suggestions"] = _latent_suggest(assessment.intent,
                                                      assessment.slots)

        # A build/project clarification commits choices that drive a whole
        # deliverable — flag it so the client shows a result-preview confirm
        # step before generating (R23).
        if build_priority and questions:
            meta["preview"] = True

        # Deterministic safety net: a build/project request with no tech named
        # MUST ask, even if the gate model hesitated and declined.
        if build_priority and not questions:
            questions = default_build_questions()
            meta = {
                **_META_DEFAULTS,
                "confidence": 0.3,
                "blocking": True,
                "reason": "These choices shape the whole project.",
                "estimated_questions_saved": 5,
                "preview": True,
            }

        # Deterministic fallback for a CODE or DOCUMENT request that omitted a
        # required choice (language / document format): if the gate model
        # declined, ask the specific question — blocking, so we ask FIRST rather
        # than silently defaulting (e.g. assuming Python, or picking a format).
        if (not build_priority and not questions
                and assessment.decision == CLARIFY):
            if ("language" in assessment.missing_required
                    and _has_code_request(question)):
                questions = language_choice_question()
                meta = {
                    **_META_DEFAULTS,
                    "confidence": 0.35,
                    "blocking": True,
                    "reason": "Tell me the language and I'll write it.",
                    "estimated_questions_saved": 2,
                }
            elif ("doc_format" in assessment.missing_required
                  and _has_doc_signal(question)):
                questions = document_choice_question()
                meta = {
                    **_META_DEFAULTS,
                    "confidence": 0.35,
                    "blocking": True,
                    "reason": "Tell me the document type and I'll generate it.",
                    "estimated_questions_saved": 2,
                }

        board.write("clarifying_questions", questions, agent=self.name)
        board.write(KEY_CLARIFY_META, meta, agent=self.name)


# Deterministic container/format words that signal an actual FILE deliverable.
# Deliberately excludes conversational nouns like "report"/"summary"/"details"
# (mirrors _DOC_RETRIEVAL_RE's exclusions) so an ordinary follow-up — "give me
# some more details on it" — is NOT mistaken for a document request.
_DOC_SIGNAL_RE = re.compile(
    r"\b(document|doc|file|download|downloadable|export|attachment|"
    r"soft\s*copy|printable|deliverable|hand-?out|write-?up)\b",
    re.I,
)


def _has_doc_signal(text: str) -> bool:
    """True iff the message shows a real document/file deliverable signal.

    Gates the deterministic "which document format?" ask so it fires only when
    the user plausibly wants a FILE — not whenever the LLM assessment guesses a
    ``doc_format`` slot on a plain follow-up or a rendering request ("in a
    tabular format"). Authority is the SEMANTIC ``document_request`` gate
    (via `explicit_doc_request`'s embedding tail — its negatives now include
    display-format asks); the deterministic detectors are the fail-open net.
    """
    t = text or ""
    try:
        from app.documents.detect import explicit_doc_request, mentions_format
        if explicit_doc_request(t)[0] or mentions_format(t):
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(_DOC_SIGNAL_RE.search(t))


# Fail-open fast-path ONLY (used when the embedder is warming/unavailable): a
# build/write verb reaching a code noun. The SEMANTIC `code_request` gate is the
# authority — this regex is never consulted while embeddings are available.
_CODE_REQUEST_RE = re.compile(
    r"\b(write|create|build|implement|generat\w*|make|develop|"
    r"code(?:\s+up)?|refactor|debug|fix|optimi[sz]e)\b[^.?!\n]{0,48}?\b("
    r"code|program|programme|function|func|method|class|script|snippet|"
    r"algorithm|app|application|api|endpoint|query|sql|regex|component|"
    r"module|cli|bot|game|website|web\s*app|webapp|server|crud|solution|"
    r"parser|scraper|schema)\b",
    re.I,
)


def _has_code_request(text: str) -> bool:
    """Is the turn genuinely asking us to WRITE code? SEMANTIC-first: the
    exemplar ``code_request`` gate is the authority (it generalizes to
    paraphrases and rejects statements/follow-ups a classifier mislabels as
    code_generation); the regex is only the fail-open fast-path used while the
    embedder is warming.

    A turn that already NAMES a language never reaches the language ask (that
    slot is suppressed upstream), so this only needs to catch the "write some
    code, unspecified language" case — and, crucially, to REJECT the misreads
    ("I don't want pin and section", "get me a tabular format")."""
    t = text or ""
    try:
        from app.semantics import gates
        verdict = gates.matches("code_request", t)
        if verdict is not None:
            return verdict
    except Exception:  # noqa: BLE001
        pass
    return bool(_CODE_REQUEST_RE.search(t))


def language_choice_question() -> list[dict]:
    """Deterministic 'which language?' card for a code request that named no
    language/framework (used when the gate model declines). One question, with a
    'You decide' escape so users who don't care dismiss it fast."""
    return [
        {
            "id": "language",
            "question": "Which language should I write it in?",
            "header": "Language",
            "kind": "single",
            "multiSelect": False,
            "reason": "The language determines the whole implementation.",
            "options": [
                {"id": "py", "label": "Python",
                 "description": "Concise — great for math/algorithms",
                 "recommended": True},
                {"id": "ts", "label": "JavaScript / TypeScript",
                 "description": "Web / Node", "recommended": False},
                {"id": "java", "label": "Java",
                 "description": "Enterprise / Android", "recommended": False},
                {"id": "decide", "label": "You decide",
                 "description": "Pick the best fit for the task",
                 "recommended": False},
            ],
        }
    ]


def archive_format_question() -> list[dict]:
    """Deterministic 'which archive format?' card for a compress/download
    request that named no format. Exactly two creatable formats — ZIP and 7z.
    (tar and rar are intentionally excluded; rar can't be created with open
    tooling, and we standardised on these two.)"""
    return [
        {
            "id": "archive_format",
            "question": "Which archive format should I create?",
            "header": "Format",
            "kind": "single",
            "multiSelect": False,
            "reason": "ZIP opens everywhere; 7z compresses smaller.",
            "options": [
                {"id": "zip", "label": "ZIP (.zip)",
                 "description": "Universal — opens everywhere",
                 "recommended": True},
                {"id": "sevenz", "label": "7-Zip (.7z)",
                 "description": "Smaller file, higher compression",
                 "recommended": False},
            ],
        }
    ]


def document_choice_question() -> list[dict]:
    """Deterministic 'which document format?' card for a document request with
    no stated format (e.g. "get me a document for this"). Offers the common
    export formats so the user picks one and we generate exactly that, instead
    of silently defaulting to PDF."""
    return [
        {
            "id": "doc_format",
            "question": "Which format should the document be?",
            "header": "Format",
            "kind": "single",
            "multiSelect": False,
            "reason": "So I generate the exact file you want, not a default.",
            "options": [
                {"id": "pdf", "label": "PDF",
                 "description": "Formatted, shareable document",
                 "recommended": True},
                {"id": "docx", "label": "Word (.docx)",
                 "description": "Editable Word document", "recommended": False},
                {"id": "md", "label": "Markdown (.md)",
                 "description": "Plain, readable Markdown", "recommended": False},
                {"id": "txt", "label": "Text (.txt)",
                 "description": "Plain text file", "recommended": False},
            ],
        }
    ]


def default_build_questions() -> list[dict]:
    """Fallback tech questions for an ambiguous build request (used only when
    the gate model declines). Kept generic-but-useful; the model normally
    produces project-specific ones."""
    return [
        {
            "id": "language",
            "question": "Which programming language should the project use?",
            "header": "Language",
            "kind": "single",
            "multiSelect": False,
            "reason": "The language shapes the whole codebase.",
            "options": [
                {"id": "py", "label": "Python",
                 "description": "Versatile, great for backends/data", "recommended": True},
                {"id": "ts", "label": "JavaScript / TypeScript",
                 "description": "Web front-end & Node", "recommended": False},
                {"id": "java", "label": "Java", "description": "Enterprise, Android",
                 "recommended": False},
                {"id": "decide", "label": "Let me decide",
                 "description": "Pick the best fit", "recommended": False},
            ],
        },
        {
            "id": "framework",
            "question": "Which framework or stack do you prefer?",
            "header": "Framework",
            "kind": "single",
            "multiSelect": False,
            "reason": "The stack affects structure and tooling.",
            "options": [
                {"id": "react", "label": "React", "description": "Web UI (TS/JS)",
                 "recommended": True},
                {"id": "next", "label": "Next.js", "description": "Full-stack React",
                 "recommended": False},
                {"id": "flutter", "label": "Flutter",
                 "description": "Cross-platform app", "recommended": False},
                {"id": "none", "label": "No preference", "description": "You choose",
                 "recommended": False},
            ],
        },
        {
            "id": "platform",
            "question": "What kind of application is this?",
            "header": "Platform",
            "kind": "single",
            "multiSelect": False,
            "reason": "The target platform changes the architecture.",
            "options": [
                {"id": "web", "label": "Web app", "description": "Runs in the browser",
                 "recommended": True},
                {"id": "desktop", "label": "Desktop app",
                 "description": "Windows/macOS/Linux", "recommended": False},
                {"id": "mobile", "label": "Mobile app", "description": "Android/iOS",
                 "recommended": False},
                {"id": "cli", "label": "CLI / script", "description": "Command line",
                 "recommended": False},
            ],
        },
    ]


def is_sample_popup_request(question: str) -> bool:
    """True when the user is asking to SEE a sample/demo clarification popup
    (no task domain), e.g. 'can you provide a sample user question popup'."""
    return bool(_SAMPLE_RE.search(question or ""))


def is_risky_operation_request(question: str) -> bool:
    """True when the request describes a clearly destructive / hard-to-reverse
    action to PERFORM (not a request for code or an explanation about it)."""
    q = question or ""
    if _CODEY_RE.search(q):
        return False
    return bool(_RISKY_RE.search(q))


def _risky_payload() -> tuple[list[dict], dict]:
    """A confirm-or-safer-alternative card for a destructive request (R15)."""
    questions = [
        {
            "id": "confirm",
            "question": "This looks like a destructive action. How should I proceed?",
            "header": "Confirm",
            "kind": "single",
            "multiSelect": False,
            "reason": "Destructive actions can be hard or impossible to undo.",
            "options": [
                {"id": "preview", "label": "Preview what would be affected first",
                 "description": "See the impact before anything changes",
                 "recommended": True},
                {"id": "script", "label": "Generate a script only",
                 "description": "Review it before running", "recommended": False},
                {"id": "proceed", "label": "Proceed as described",
                 "description": "Continue with the full request", "recommended": False},
            ],
        }
    ]
    meta = {
        **_META_DEFAULTS,
        "confidence": 0.35,
        "blocking": True,
        "reason": "This looks destructive and hard to reverse.",
        "estimated_questions_saved": 1,
    }
    return questions, meta


def _sample_payload() -> tuple[list[dict], dict]:
    """A generic, domain-free demo popup (R21)."""
    questions = [
        {
            "id": "demo1", "question": "Which option do you prefer?",
            "header": "Option", "kind": "single", "multiSelect": False,
            "reason": "Demonstrates a single-select question.",
            "options": [
                {"id": "a", "label": "Option A", "description": "", "recommended": True},
                {"id": "b", "label": "Option B", "description": "", "recommended": False},
                {"id": "c", "label": "Option C", "description": "", "recommended": False},
            ],
        },
        {
            "id": "demo2", "question": "How important is speed?",
            "header": "Speed", "kind": "single", "multiSelect": False,
            "reason": "Demonstrates a priority question.",
            "options": [
                {"id": "low", "label": "Low", "description": "", "recommended": False},
                {"id": "med", "label": "Medium", "description": "", "recommended": True},
                {"id": "high", "label": "High", "description": "", "recommended": False},
            ],
        },
    ]
    meta = {
        **_META_DEFAULTS,
        "confidence": 0.5,
        "blocking": False,
        "reason": "This is a sample clarification popup.",
        "mode": "sample",
        "sample": True,
    }
    return questions, meta


_MODE_DIRECTIVES = {
    "explorer": "Mode: Explorer — the user welcomes thorough clarification; "
                "ask the fullest set the band allows (still <=3).",
    "builder": "Mode: Builder — the user wants to move fast; ask at most ONE "
               "question and only if it truly blocks a good answer.",
    "expert": "Mode: Expert — the user is technical; restrict questions to "
              "technical/constraint choices, never experience level or audience.",
    "autopilot": "Mode: Autopilot — only ask when you are genuinely lost; "
                 "otherwise proceed with your best assumptions.",
    "teacher": "Mode: Teacher — for explanations/tutorials, first ask the "
               "user's learning goal or experience level.",
}


def _apply_mode(questions: list[dict], meta: dict, mode: str) -> list[dict]:
    """Apply the active clarification personality on top of the band (R18).

    Deterministic clamps: Autopilot only asks below 0.40 confidence; Builder
    asks at most one question. Explorer/Expert/Teacher shape CONTENT via the
    prompt, so they keep the band's question count here."""
    if not questions or not mode:
        return questions
    try:
        c = float(meta.get("confidence", 1.0))
    except (TypeError, ValueError):
        c = 1.0
    if mode == "autopilot":
        return questions if c < _band_targeted() else []
    if mode == "builder":
        return questions[:1]
    return questions


def _apply_band(questions: list[dict], meta: dict,
                high_band: float | None = None) -> list[dict]:
    """Clamp the question set to the confidence band (R2). Sample popups bypass
    banding. High confidence → answer (drop questions). `high_band` is the
    answer threshold; the clarifier lowers it under fatigue/eroded trust
    (advanced-intent-reasoning R3/R4) so the system asks less — safety cards
    never reach here, so they are never suppressed."""
    if high_band is None:
        high_band = _band_high()
    if meta.get("mode") == "sample":
        return questions
    if not questions:
        return questions
    try:
        c = float(meta.get("confidence", 1.0))
    except (TypeError, ValueError):
        c = 1.0
    if c > high_band:
        return []                      # answer directly
    if c >= _band_assume():
        return questions[:1]           # assumption band — minimal ask
    if c >= _band_targeted():
        return questions[:2]           # targeted
    return questions[:_MAX_QUESTIONS]  # guided


def _parse_questions(raw: str) -> list[dict]:
    """Back-compat wrapper: parse only the question list from the model JSON.

    Preserved for the existing contract tests; new callers use [_parse_payload].
    """
    return _parse_payload(raw)[0]


def _parse_payload(raw: str) -> tuple[list[dict], dict]:
    """Parse + sanitize the model's JSON into (questions, clarify_meta).

    Tolerant of code fences / stray prose. Returns ([], defaults) for anything
    malformed or when the model declined to clarify. Every meta field is
    defaulted so a missing/garbled value never breaks the turn (R3.4).
    """
    meta = dict(_META_DEFAULTS)
    obj = _loads_loose(raw)
    if not isinstance(obj, dict):
        return [], meta

    # ---- meta (parsed even when declining, so confidence is available) ----
    meta["confidence"] = _coerce_confidence(obj.get("confidence"),
                                            has_questions=bool(obj.get("clarify")))
    meta["blocking"] = bool(obj.get("blocking", False))
    meta["reason"] = str(obj.get("reason") or "").strip()[:200]
    meta["estimated_questions_saved"] = _coerce_int(
        obj.get("estimated_questions_saved"), lo=0, hi=99)
    mode = str(obj.get("mode") or "ask").strip().lower()
    meta["mode"] = mode if mode in _VALID_MODES else "ask"
    meta["assumptions"] = _parse_assumptions(obj.get("assumptions"))
    meta["preview"] = bool(obj.get("preview", False))
    # advanced-intent-reasoning R5: optional interpretation set (same call).
    meta["interpretations"] = parse_interpretations(obj.get("interpretations"))

    if not obj.get("clarify", False):
        return [], meta
    raw_qs = obj.get("questions")
    if not isinstance(raw_qs, list):
        return [], meta

    out: list[dict] = []
    for i, q in enumerate(raw_qs[:_MAX_QUESTIONS]):
        if not isinstance(q, dict):
            continue
        text = str(q.get("question") or "").strip()
        if not text:
            continue
        raw_opts = q.get("options")
        if not isinstance(raw_opts, list):
            continue
        options: list[dict] = []
        for j, o in enumerate(raw_opts[:_MAX_OPTIONS]):
            if not isinstance(o, dict):
                continue
            label = str(o.get("label") or "").strip()
            if not label:
                continue
            options.append(
                {
                    "id": str(o.get("id") or "").strip() or f"o{j + 1}",
                    "label": label,
                    "description": str(o.get("description") or "").strip(),
                    "recommended": bool(o.get("recommended")),
                }
            )
        if len(options) < 2:
            continue  # a clarifying question needs real choices
        # At most ONE recommended option per question (keep the first).
        seen_rec = False
        for opt in options:
            if opt["recommended"] and not seen_rec:
                seen_rec = True
            else:
                opt["recommended"] = False
        # Trust the model's semantic 'kind' choice (no keyword overrides);
        # tolerate the older `multiSelect` bool only when 'kind' is missing.
        kind = str(q.get("kind") or "").strip().lower()
        if kind not in _VALID_KINDS:
            kind = "multi" if q.get("multiSelect") else "single"
        out.append(
            {
                "id": str(q.get("id") or "").strip() or f"q{i + 1}",
                "question": text,
                "header": str(q.get("header") or "").strip()[:16],
                "kind": kind,
                # Kept for any older client; True only for multi.
                "multiSelect": kind == "multi",
                "reason": str(q.get("reason") or "").strip()[:200],
                "options": options,
            }
        )
    return out, meta


def _coerce_confidence(value, *, has_questions: bool) -> float:
    """0.0..1.0 confidence. Out-of-range/unparseable → 1.0 when declining, or a
    mid value (0.5) when the model is asking (so band logic keeps the ask)."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.5 if has_questions else 1.0
    if c < 0.0 or c > 1.0:
        return 0.5 if has_questions else 1.0
    return c


def _coerce_int(value, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, n))


def _parse_assumptions(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for i, a in enumerate(raw):
        if not isinstance(a, dict):
            continue
        label = str(a.get("label") or "").strip()
        if not label:
            continue
        out.append({
            "id": str(a.get("id") or "").strip() or f"a{i + 1}",
            "label": label,
            "value": str(a.get("value") or "").strip(),
        })
    return out


def _loads_loose(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    # Strip ```json … ``` fences if present.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    # Last resort: grab the outermost {...}.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None
