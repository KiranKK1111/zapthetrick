"""Situational intelligence + two-lane display (vNext §4.15, Stage 7 Component J).

Beyond WHAT the interviewer asks is HOW they ask it. The same question inside a
stress probe, a harsh challenge, a conviction trap ("are you SURE?"), a salary
round, or a rapport moment demands a different posture — and the band matters
(an intern holding firm reads differently from a principal doing so). §4.15:

  * a **situation classifier** over the interviewer channel — SEMANTIC first
    (`app/semantics/gates.classify`, generalizes to paraphrase), the cue lists
    only a cold-start fallback — optionally fused with the acoustic emotion
    signal (`app/live/emotion.py`);
  * per-situation × band **strategies** — a `directive` that shades the
    DICTATABLE answer, plus GUIDANCE **whisper chips** (amber, never read aloud:
    "they're testing conviction — hold your ground", "slow down");
  * the **two-lane display contract** — DICTATABLE (spoken) vs GUIDANCE
    (whisper) — with a **validator** that enforces the separation so a coaching
    chip can NEVER leak into what the candidate reads out.

The salary round's fact-based strategy already lives in `app/live/negotiate.py`;
this classifies the situation and routes to it. Semantic-first + deterministic
fallback + fail-open. Flag-gated (`live.situational`, default OFF → no shading).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- situation taxonomy ---------------------------------------------------
STRESS = "stress"                 # pressure / rapid-fire / "you have 2 minutes"
HARSHNESS = "harshness"           # dismissive / hostile / "that's just wrong"
CONVICTION_TRAP = "conviction_trap"  # "are you SURE?" — testing whether you fold
SALARY = "salary"                 # comp / negotiation round
RAPPORT = "rapport"               # warm / small-talk / "tell me about yourself"
NEUTRAL = "neutral"

_SITUATIONS = (STRESS, HARSHNESS, CONVICTION_TRAP, SALARY, RAPPORT)

# Semantic exemplars per situation — the PRIMARY signal (embedded + matched by
# gates.classify, so paraphrases generalize). NOT a keyword table.
_EXEMPLARS: dict[str, list[str]] = {
    STRESS: [
        "you have two minutes, go",
        "quick, what's the answer",
        "we're short on time, be fast",
        "just give me the number now",
    ],
    HARSHNESS: [
        "that's completely wrong",
        "that's a terrible answer",
        "did you even read the question",
        "I'm not impressed, that's basic",
    ],
    CONVICTION_TRAP: [
        "are you sure about that",
        "really? you'd stake your answer on that",
        "most people say the opposite, still confident",
        "are you certain, think again",
    ],
    SALARY: [
        "what are your salary expectations",
        "what's your current compensation",
        "that's above our budget, can you go lower",
        "what number are you looking for",
    ],
    RAPPORT: [
        "tell me a bit about yourself",
        "how has your day been",
        "that's a great background, tell me more",
        "what do you enjoy outside work",
    ],
}

# Cold-start FALLBACK cues only (used when the semantic classifier is unavailable
# — disabled/no embedder). The semantic path above is authoritative.
_FALLBACK_CUES: dict[str, tuple[str, ...]] = {
    STRESS: ("two minutes", "quickly", "hurry", "short on time", "fast", "right now"),
    HARSHNESS: ("wrong", "terrible", "not impressed", "did you even", "basic", "bad answer"),
    CONVICTION_TRAP: ("are you sure", "really?", "certain", "sure about", "think again", "stake"),
    SALARY: ("salary", "compensation", "ctc", "budget", "expectations", "package", "how much"),
    RAPPORT: ("about yourself", "your day", "outside work", "tell me more", "enjoy"),
}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "situational", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class SituationRead:
    situation: str = NEUTRAL
    confidence: float = 0.0
    source: str = "none"          # "semantic" | "fallback" | "emotion" | "none"

    def to_dict(self) -> dict:
        return {"situation": self.situation, "confidence": round(self.confidence, 3),
                "source": self.source}


def _fallback_classify(text: str) -> "tuple[str, float]":
    t = (text or "").lower()
    if not t.strip():
        return NEUTRAL, 0.0
    best, best_hits = NEUTRAL, 0
    for sit, cues in _FALLBACK_CUES.items():
        hits = sum(1 for c in cues if c in t)
        if hits > best_hits:
            best, best_hits = sit, hits
    if best_hits == 0:
        return NEUTRAL, 0.0
    return best, min(1.0, 0.5 + 0.15 * best_hits)


def classify_situation(interviewer_text: str, *, emotion_label: str | None = None,
                       classify_fn=None, threshold: float = 0.55) -> SituationRead:
    """Classify the interviewer situation. SEMANTIC first (`gates.classify` over
    the exemplars — injectable via `classify_fn` for tests), cue fallback when
    the embedder is unavailable, and the acoustic `emotion_label` as a tie-break
    nudge toward STRESS/HARSHNESS. Never raises → NEUTRAL."""
    try:
        text = (interviewer_text or "").strip()
        if not text:
            # No text but a stressed voice still reads as a stress situation.
            if emotion_label in ("stressed", "rushed"):
                return SituationRead(STRESS, 0.4, "emotion")
            return SituationRead(NEUTRAL, 0.0, "none")

        # 1) Semantic (primary).
        cls = None
        try:
            fn = classify_fn
            if fn is None:
                from app.semantics import gates
                fn = lambda q: gates.classify(  # noqa: E731
                    q, _EXEMPLARS, threshold=threshold,
                    cache_key="live_situation")
            cls = fn(text)
        except Exception:  # noqa: BLE001
            cls = None
        if cls in _SITUATIONS:
            return SituationRead(cls, 0.8, "semantic")

        # 2) Cue fallback (cold start).
        sit, conf = _fallback_classify(text)
        if sit != NEUTRAL:
            return SituationRead(sit, conf, "fallback")

        # 3) Emotion-only nudge.
        if emotion_label in ("stressed", "rushed"):
            return SituationRead(STRESS, 0.4, "emotion")
        return SituationRead(NEUTRAL, 0.0, "none")
    except Exception:  # noqa: BLE001
        return SituationRead(NEUTRAL, 0.0, "none")


# ---- per-situation × band strategy ----------------------------------------
@dataclass
class GuidanceChip:
    """An amber whisper chip — GUIDANCE for the candidate, NEVER read aloud.
    `spoken` is invariant False (the validator enforces it)."""
    text: str
    kind: str = "situation"       # situation | pace | conviction | rapport
    spoken: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "spoken": False}


@dataclass
class SituationStrategy:
    situation: str
    directive: str = ""                       # shades the DICTATABLE answer
    chips: list[GuidanceChip] = field(default_factory=list)  # GUIDANCE lane

    def to_dict(self) -> dict:
        return {"situation": self.situation, "directive": self.directive,
                "chips": [c.to_dict() for c in self.chips]}


def _band_tier(band: str) -> str:
    """Collapse a band slug to junior / mid / senior for strategy shading."""
    b = (band or "").strip().lower()
    if b in ("intern", "fresher", "junior", "entry"):
        return "junior"
    if b in ("senior", "lead", "staff", "principal", "distinguished"):
        return "senior"
    return "mid"


# Directive shading per (situation) — the band modulates firmness/register.
def strategy_for(read: SituationRead | str, band: str = "") -> SituationStrategy:
    """Map a situation (+ band) to a DICTATABLE directive + GUIDANCE whisper
    chips. The directive shades the spoken answer; the chips are coaching the
    candidate reads but never says. Never raises → a neutral strategy."""
    try:
        sit = read.situation if isinstance(read, SituationRead) else str(read or NEUTRAL)
        tier = _band_tier(band)
        if sit == STRESS:
            return SituationStrategy(sit,
                directive=("Lead with the answer in the first sentence, then one "
                           "line of justification — no preamble."),
                chips=[GuidanceChip("Time pressure — give the headline first, elaborate only if asked.", "pace")])
        if sit == HARSHNESS:
            firmness = ("Acknowledge briefly, then restate your reasoning with "
                        "evidence — do not over-apologize or cave.")
            if tier == "junior":
                firmness = ("Stay calm; concede the specific point if it's fair, "
                            "then show your corrected reasoning.")
            return SituationStrategy(sit, directive=firmness,
                chips=[GuidanceChip("Harsh tone — it's a pressure test. Stay even, defend with evidence.", "situation")])
        if sit == CONVICTION_TRAP:
            hold = ("Hold your position and give the concrete reason it's "
                    "correct; only revise if they present a real counter-fact.")
            if tier == "junior":
                hold = ("Restate your reasoning confidently; if unsure, say what "
                        "would change your answer rather than flip-flopping.")
            return SituationStrategy(sit, directive=hold,
                chips=[GuidanceChip("Conviction trap — they're testing if you fold. Hold your ground unless given a real counter.", "conviction")])
        if sit == SALARY:
            return SituationStrategy(sit,
                directive=("Anchor on value and a researched range; stay "
                           "collaborative, never a fixed ultimatum."),
                chips=[GuidanceChip("Salary round — anchor to a range, tie to impact, keep it collaborative.", "situation")])
        if sit == RAPPORT:
            return SituationStrategy(sit,
                directive=("Warm and concise; a genuine, brief answer, then steer "
                           "back toward relevant strengths."),
                chips=[GuidanceChip("Rapport moment — be warm and brief, then bridge to a strength.", "rapport")])
        return SituationStrategy(NEUTRAL, directive="", chips=[])
    except Exception:  # noqa: BLE001
        return SituationStrategy(NEUTRAL, directive="", chips=[])


# ---- two-lane display contract + separation validator ---------------------
@dataclass
class TwoLaneDisplay:
    dictatable: str                            # the spoken answer
    guidance: list[GuidanceChip] = field(default_factory=list)  # whisper chips

    def to_dict(self) -> dict:
        return {"dictatable": self.dictatable,
                "guidance": [c.to_dict() for c in self.guidance]}


def validate_separation(display: TwoLaneDisplay) -> "tuple[bool, list[str]]":
    """Enforce the DICTATABLE/GUIDANCE contract: (1) every guidance chip is
    non-spoken; (2) no guidance chip's text has leaked into the dictatable
    stream (what the candidate reads aloud). Returns (ok, violations)."""
    violations: list[str] = []
    try:
        spoken = (display.dictatable or "").lower()
        for chip in display.guidance or ():
            if getattr(chip, "spoken", False):
                violations.append(f"guidance chip marked spoken: {chip.text!r}")
            ctext = (chip.text or "").strip().lower()
            if ctext and ctext in spoken:
                violations.append(f"guidance leaked into dictatable: {chip.text!r}")
    except Exception:  # noqa: BLE001
        return True, []       # fail-open: never block delivery on a validator error
    return (not violations), violations


def build_display(dictatable: str, chips) -> TwoLaneDisplay:
    """Assemble a two-lane display, DEFENSIVELY enforcing separation: every chip
    is coerced non-spoken, and any chip whose text already appears in the spoken
    answer is dropped from the guidance lane (it isn't guidance if it's said).
    Never raises."""
    try:
        spoken = (dictatable or "").strip()
        low = spoken.lower()
        clean: list[GuidanceChip] = []
        seen: set[str] = set()
        for c in chips or ():
            text = (getattr(c, "text", "") or "").strip()
            if not text or text.lower() in low or text.lower() in seen:
                continue
            seen.add(text.lower())
            kind = getattr(c, "kind", "situation")
            clean.append(GuidanceChip(text=text, kind=kind, spoken=False))
        return TwoLaneDisplay(dictatable=spoken, guidance=clean)
    except Exception:  # noqa: BLE001
        return TwoLaneDisplay(dictatable=(dictatable or "").strip(), guidance=[])


__all__ = ["STRESS", "HARSHNESS", "CONVICTION_TRAP", "SALARY", "RAPPORT", "NEUTRAL",
           "SituationRead", "classify_situation", "GuidanceChip", "SituationStrategy",
           "strategy_for", "TwoLaneDisplay", "validate_separation", "build_display",
           "enabled"]
