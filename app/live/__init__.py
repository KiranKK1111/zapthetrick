"""
Live conversational intelligence (live-conversational-intelligence spec).

Phase 1 — structured events + interview state machine + turn-taking. Every
module here is additive, flag-gated (`cfg.live.*`), and fail-open: with the
flags off the Live module behaves byte-for-byte as today
(transcribe -> agent.predict -> answer). No new DB schema (per-session state is
in-process); no second blocking LLM call (the event typer reuses the single
`question_detection.agent.predict` call).
"""
