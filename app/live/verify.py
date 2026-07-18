"""
Live answer Verifier — the post-LLM stage the live path never had.

After an answer finishes streaming, a FAST verification call scores it:

    relevance          does it answer the detected question?
    hallucination_risk unsupported/overconfident claims?
    verdict            "ok" | "weak"

The score is emitted as additive meta (`verify` on the answer's qid) so the
UI can badge the answer, and published on the session event bus
(ANSWER_VERIFIED). When the verdict is weak AND `cfg.live.answer_regenerate`
is on, the caller regenerates once with the verifier's critique folded into
the directive.

Deliberately NON-BLOCKING: verification runs after the tokens have already
streamed — it never adds latency to the visible answer. Fail-open: any
error → no verdict, no regen.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.core.config_loader import cfg

log = logging.getLogger(__name__)

_PROMPT = """You are verifying an interview assistant's answer. Score it strictly.

QUESTION: {question}
TOPIC: {topic}
ANSWER:
{answer}

Reply with ONLY compact JSON:
{{"relevance": 0.0-1.0, "hallucination_risk": 0.0-1.0, "issue": "<=12 words or empty"}}"""


@dataclass
class Verdict:
    relevance: float
    hallucination_risk: float
    issue: str
    # True when the answer is structurally garbled (deterministic check below) —
    # forces a non-ok verdict regardless of the semantic scores.
    gibberish: bool = False
    # True when the answer LEAKED the model's internal reasoning / planning /
    # system-prompt / continuation re-prompt instead of the spoken answer.
    leaked: bool = False

    @property
    def ok(self) -> bool:
        if self.gibberish or self.leaked:
            return False
        threshold = float(getattr(cfg.live, "verify_min_relevance", 0.55) or 0.55)
        risk_cap = float(getattr(cfg.live, "verify_max_hallucination", 0.75) or 0.75)
        return self.relevance >= threshold and self.hallucination_risk <= risk_cap

    def to_meta(self) -> dict:
        return {
            "relevance": round(self.relevance, 2),
            "hallucination_risk": round(self.hallucination_risk, 2),
            "verdict": "ok" if self.ok else "weak",
            **({"gibberish": True} if self.gibberish else {}),
            **({"leaked": True} if self.leaked else {}),
            **({"issue": self.issue} if self.issue else {}),
        }


# Prompt-internal strings that must NEVER appear in a spoken answer — their
# presence anywhere means the model dumped the prompt / its own framing.
_LEAK_HARD = (
    "interviewer question:", "candidate level:", "fast factual mode",
    "answer guidance", "elite interview answer", "## interviewer", "## answer",
    "thinking process", "output only the continuation",
    "the previous reply ended", "the previous text ended",
    "resume from exactly where", "joins seamlessly onto the existing text",
)
# Meta-reasoning openers — a real answer never STARTS with these. Checked only
# against the head of the answer (leaks always begin at the very start).
_LEAK_HEAD = (
    "we need to answer", "we need to produce", "we need to continue",
    "we need to resume", "we need to output", "we need to write",
    "we should output", "we should produce", "we must follow",
    "we must answer", "we can craft", "let's craft", "let me craft",
    "analyze the request", "analyze the question", "the user is asking",
    "the user wants", "the interviewer is asking", "according to guidance",
    "given the instructions", "given the interview context",
)


def looks_like_leaked_reasoning(answer: str) -> bool:
    """True when the answer is the model's internal reasoning / planning /
    system-prompt / continuation re-prompt rather than the actual answer — the
    coherent-but-wrong failure the semantic relevance score misses (seen in the
    exported live sessions: "We need to answer…", "Thinking Process:",
    "We need to continue from where the previous reply stopped…"). Never raises."""
    try:
        t = (answer or "").strip().lower()
        if not t:
            return False
        if any(m in t for m in _LEAK_HARD):
            return True
        head = t[:220]
        return any(m in head for m in _LEAK_HEAD)
    except Exception:  # noqa: BLE001
        return False


def looks_incoherent(answer: str) -> bool:
    """Deterministic structural-garbage check on a FINISHED answer — catches
    broken output the semantic relevance score can miss: unknown/replacement
    tokens, whitespace-free mashes, or one token dominating the text (runaway
    repetition). Cheap; runs before the LLM verifier. Never raises."""
    try:
        t = (answer or "").strip()
        if not t:
            return True
        if "<unk>" in t or t.count("�") >= 3:
            return True
        if max((len(w) for w in t.split()), default=0) >= 50:
            return True
        if len(t) >= 400:
            ws = sum(1 for c in t if c.isspace())
            if ws / len(t) < 0.03:
                return True
        words = re.findall(r"\w+", t.lower())
        if len(words) >= 20:
            from collections import Counter
            _, top_n = Counter(words).most_common(1)[0]
            if top_n / len(words) > 0.4:      # one word is >40% of the answer
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _parse(text: str) -> Verdict | None:
    """Extract the JSON verdict from a model reply (tolerates fences/prose)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return Verdict(
            relevance=max(0.0, min(1.0, float(obj.get("relevance", 0.0)))),
            hallucination_risk=max(0.0, min(1.0, float(
                obj.get("hallucination_risk", 1.0)))),
            issue=str(obj.get("issue") or "")[:120],
        )
    except Exception:  # noqa: BLE001
        return None


async def verify_answer(question: str, answer: str, topic: str = "",
                        *, session_key: str | None = None) -> Verdict | None:
    """Score one finished answer with a fast routed completion. None on any
    failure (fail-open — verification is a quality layer, not a gate)."""
    if not (question or "").strip() or not (answer or "").strip():
        return None
    # Structural garbage / reasoning leaks are unambiguous — flag them
    # deterministically and skip the (pointless) semantic scoring call so the
    # retry fires immediately.
    if looks_incoherent(answer):
        return Verdict(relevance=0.0, hallucination_risk=1.0,
                       issue="garbled / incoherent output", gibberish=True)
    if looks_like_leaked_reasoning(answer):
        return Verdict(relevance=0.0, hallucination_risk=1.0,
                       issue="leaked internal reasoning / prompt instead of the answer",
                       leaked=True)
    try:
        # Same client the question-detection agent uses (provider-aware:
        # auto-routing, ollama, or an OpenAI-compatible endpoint) — the
        # low-level router alone can't serve deployments without a DB
        # fallback chain configured.
        from app.core.llm_client import llm
        prompt = _PROMPT.format(
            question=question.strip()[:600],
            topic=(topic or "general").strip()[:80],
            # Verify the head of a long answer; tail truncation doesn't change
            # whether the answer addressed the question.
            answer=answer.strip()[:2400],
        )
        text = await llm.complete(
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "max_tokens": 160},
        )
        return _parse(text)
    except Exception as exc:  # noqa: BLE001
        log.info("live verify failed: %s", exc)
        return None


def critique_directive(verdict: Verdict, question: str) -> str:
    """Directive for a regeneration pass on a weak/garbled verdict."""
    bits = ["Your previous answer to this question scored low on review —"]
    if verdict.leaked:
        bits.append("it leaked internal planning / meta-commentary / prompt text "
                    "into the reply. Output ONLY the final spoken answer — NO "
                    "'we need to…', 'thinking process', outline planning, "
                    "restating the instructions, or 'continue from where we left "
                    "off'. Start directly with the answer;")
    if verdict.gibberish:
        bits.append("it came out garbled / incoherent — write a clean, "
                    "well-structured answer in clear prose;")
    if verdict.relevance < 0.55:
        bits.append("it did not directly answer what was asked;")
    if verdict.hallucination_risk > 0.75:
        bits.append("it made claims that may not be supportable — hedge or drop them;")
    if verdict.issue:
        bits.append(f"reviewer note: {verdict.issue}.")
    bits.append(f'Answer the question directly and concretely: "{question.strip()[:300]}"')
    return " ".join(bits)
