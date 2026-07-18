"""
Persona-answer tool: shape the final reply in the candidate's voice.

This is the last step of an answer flow. It takes whatever raw context
the previous tools gathered (resume hits, web snippets, code result),
plus the question, and asks the LLM to produce a polished first-person
answer using `persona/voice.py`'s template.

It's invoked by the orchestrator after any data-gathering tool calls.
The result is what the user sees.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator

from app.core.config_loader import cfg
from app.core.llm_client import llm
from app.persona.voice import (
    build_candidate_prompt,
    build_interview_answer_prompt,
    build_live_answer_prompt,
    build_profile_answer_prompt,
)
from app.tools.registry import Tool, register


INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "profile": {"type": "object"},
        "context": {
            "type": "string",
            "description": "Free-form context (resume hits, web snippets, ...).",
        },
        "prior_qa": {
            "type": "string",
            "description": "Most recent Q+A, included on follow-up questions.",
        },
    },
    "required": ["question", "profile"],
}


def _build_messages(
    *,
    question: str,
    profile: dict,
    context: str | None,
    prior_qa: str | None,
    qtype: str = "unknown",
    concise: bool = False,
    directive: str | None = None,
    profile_q: bool = False,
) -> list[dict[str, str]]:
    # Deep, chat-quality interview answer (structured Markdown, code, diagrams)
    # specialised by question type. `concise` swaps in the terse real-time
    # prompt (only when explicitly requested — the live path now defaults to
    # the detailed prompt so live answers match chat quality). Falls back to
    # the short first-person candidate voice only if qtype="candidate".
    # `profile_q` + a profile → the spoken dictate-ready prompt: questions
    # about the candidate themselves are read aloud verbatim, so they must be
    # crisp first-person speech, not a formatted technical essay.
    if qtype == "candidate":
        system = build_candidate_prompt(profile)
    elif profile_q and profile:
        system = build_profile_answer_prompt(profile)
    elif concise:
        system = build_live_answer_prompt(profile, qtype)
    else:
        system = build_interview_answer_prompt(profile, qtype)
    user_parts: list[str] = []
    # Live deliberation directive (strategy scaffold + plan + hedge): an
    # additive ANSWER GUIDANCE block — shapes the answer within this same call.
    if directive and directive.strip():
        user_parts.append("ANSWER GUIDANCE (shape your answer accordingly):\n" + directive.strip())
    if prior_qa:
        user_parts.append(
            "CONVERSATION SO FAR (earlier interviewer questions and your answers "
            "in this session, oldest first — the current question may be a "
            "follow-up that refers back to these via \"it\", \"that\", \"those\", "
            "etc.; resolve such references against this thread):\n" + prior_qa
        )
    if context:
        user_parts.append(
            "ADDITIONAL CONTEXT (resume hits / web snippets — use only what's relevant):\n"
            + context
        )
    user_parts.append("INTERVIEWER QUESTION:\n" + question)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


async def answer(
    *,
    question: str,
    profile: dict,
    context: str | None = None,
    prior_qa: str | None = None,
    qtype: str = "unknown",
    difficulty: str = "standard",
    profile_q: bool = False,
) -> str:
    """Non-streaming variant. Returns the full answer as one string."""
    messages = _build_messages(
        question=question, profile=profile, context=context,
        prior_qa=prior_qa, qtype=qtype, profile_q=profile_q,
    )
    parts: list[str] = []
    async for chunk in llm.stream_chat(messages, options={"difficulty": difficulty}):
        parts.append(chunk)
    return "".join(parts).strip()


async def stream(
    *,
    question: str,
    profile: dict,
    context: str | None = None,
    prior_qa: str | None = None,
    qtype: str = "unknown",
    model: str | None = None,
    difficulty: str = "standard",
    concise: bool = False,
    max_tokens: int | None = None,
    directive: str | None = None,
    profile_q: bool = False,
) -> AsyncGenerator[str, None]:
    """Streaming variant. Used by SSE / WebSocket endpoints. `model` is a fast
    first-token PREFERENCE for the live path (cfg.llm.live_model); the auto
    router still falls back through every model + escalates hard/expert.
    `concise` uses the terse real-time prompt (default off — live answers are
    now detailed like chat); `max_tokens` caps the answer length; `directive`
    injects an additive live-deliberation guidance block into the same call."""
    messages = _build_messages(
        question=question, profile=profile, context=context,
        prior_qa=prior_qa, qtype=qtype, concise=concise, directive=directive,
        profile_q=profile_q,
    )
    options: dict = {"difficulty": difficulty}
    if max_tokens:
        options["max_tokens"] = max_tokens
    async for chunk in llm.stream_chat(
        messages, model=model or None, options=options
    ):
        yield chunk


register(Tool(
    name="persona_answer",
    description=(
        "Render the final answer in the candidate's first-person voice. "
        "Call this last, after any data-gathering tools."
    ),
    input_schema=INPUT_SCHEMA,
    handler=answer,
))
