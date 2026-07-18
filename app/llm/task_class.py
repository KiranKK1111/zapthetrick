"""Prompt-type classifier (intelligent-model-routing R2).

`classify_task(text, intent, difficulty) -> Task_Category` maps a request to one
of ``capabilities.TASK_CATEGORIES`` by reusing the EXISTING deterministic intent
signal (`intent_pipeline.detect_intent`) plus small category lexicons — never a
second blocking LLM call (R2.1/R11.1). Unknown → ``general`` (R2.2), and the
result is deterministic for deterministic inputs (R2.3, Property 3).
"""
from __future__ import annotations

import re

from app.llm.capabilities import TASK_CATEGORIES

# Intent (from intent_pipeline) → Task_Category. Intents not listed fall through
# to the lexical cues, then to `general`.
_INTENT_MAP = {
    "code_generation": "coding",
    "project_build": "agentic",
    "debugging": "coding",
    "test_generation": "coding",
    "design": "architecture",
    "documentation": "writing",
    "comparison": "reasoning",
    "knowledge": "research",
    "chitchat": "conversation",
}

# Lexical cues checked when the intent is generic/unknown. Most-specific first.
_CUES = [
    ("math", re.compile(
        r"\b(integral|derivative|equation|theorem|proof|matrix|probability|"
        r"algebra|calculus|solve for|factori[sz]e)\b", re.I)),
    ("coding", re.compile(
        r"\b(code|function|bug|refactor|compile|api|regex|class|method|"
        r"stack ?trace|exception|unit test|python|javascript|typescript|rust|"
        r"java|c\+\+|sql)\b", re.I)),
    ("architecture", re.compile(
        r"\b(architect|design (a|the|this|my) (system|app|service)|scalab|"
        r"microservice|schema|data model|trade-?off|system design)\b", re.I)),
    ("vision", re.compile(r"\b(image|screenshot|photo|picture|diagram|chart)\b", re.I)),
    ("research", re.compile(
        r"\b(research|compare|summari[sz]e|explain|what is|how does|why does|"
        r"sources?|cite|literature)\b", re.I)),
    ("writing", re.compile(
        r"\b(write|draft|essay|article|blog|email|letter|story|rephrase|"
        r"proofread|document|readme)\b", re.I)),
    ("reasoning", re.compile(
        r"\b(reason|think step|logic|deduce|infer|puzzle|plan|strateg)\b", re.I)),
    ("agentic", re.compile(
        r"\b(build (an?|the|my) (app|project|tool)|scaffold|implement (a|the) "
        r"(project|app)|create (an?|the) (app|project))\b", re.I)),
]


def classify_task(text: str, intent: str | None = None,
                  difficulty: str | None = None) -> str:
    """Return a Task_Category. Deterministic; fail-open to ``general``."""
    try:
        return _classify(text, intent, difficulty)
    except Exception:  # noqa: BLE001
        return "general"


def _classify(text: str, intent: str | None, difficulty: str | None) -> str:
    # 1) Trust the existing intent signal first.
    label = _INTENT_MAP.get((intent or "").strip().lower())
    if label in TASK_CATEGORIES:
        return label

    t = text or ""
    if not t.strip():
        return "general"

    # 2) Lexical cues (most-specific first).
    for cat, rx in _CUES:
        if rx.search(t):
            return cat

    # 3) A vision turn signalled via difficulty/intent already mapped above;
    #    otherwise a hard/expert turn with no cue leans reasoning, else general.
    if (difficulty or "").lower() in ("hard", "expert"):
        return "reasoning"
    return "general"


__all__ = ["classify_task"]
