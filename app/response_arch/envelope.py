"""Unified response envelope — `response.v1` (Architecture.md §5).

One canonical, versioned object per turn. SSE streams it progressively; the same
object is (will be) persisted with the message so *live == reload*. This module
defines the schema + a builder that assembles it from the fields a turn already
computes, so the `done` event can carry the whole envelope additively without
changing the existing streaming contract.

Additive & fail-open: unknown/absent fields are simply omitted (`exclude_none`),
so old clients ignore it and new clients read whatever is present.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResponseEnvelope(BaseModel):
    """The `response.v1` contract. See Architecture.md §5 for field semantics."""
    model_config = ConfigDict(populate_by_name=True)

    # `schema` is a reserved-ish name on BaseModel, so store as `schema_` and
    # serialize under the alias "schema".
    schema_: str = Field(default="response.v1", alias="schema")
    conversation_id: str | None = None
    message_id: str | None = None

    # multimodal input summary (Architecture §12) — {modality, text, images,
    # files, audio}. The raw bytes live in `Message.sources`; this is a compact,
    # reload-stable descriptor of what the user sent this turn.
    input: dict | None = None

    # what we understood
    intent: dict | None = None                 # {type, confidence, source, secondary}
    difficulty: str | None = None
    resolved_prompt: str | None = None
    topic_shift: bool = False

    # the answer (text is streamed via `token`; here we carry its shape/state)
    answer: dict = Field(default_factory=lambda: {"incomplete": False})

    # follow-up suggestion chips (also streamed; included when known)
    suggestions: list[dict] = Field(default_factory=list)

    # optional, populated by the intent profile / mesh
    document: dict | None = None               # only when explicitly requested
    artifacts: list[dict] = Field(default_factory=list)
    knowledge: dict | None = None              # {related: [{entity, relation?}]}
    grounding: dict | None = None
    clarification: dict | None = None

    # quality + telemetry
    meta: dict = Field(default_factory=dict)

    def as_json(self) -> dict:
        """Canonical JSON (alias keys, no None fields)."""
        return self.model_dump(by_alias=True, exclude_none=True)


def build_envelope(
    *,
    conversation_id: str | None = None,
    message_id: str | None = None,
    intent: dict | None = None,
    difficulty: str | None = None,
    resolved_prompt: str | None = None,
    topic_shift: bool = False,
    answer_shape: str | None = None,
    incomplete: bool = False,
    suggestions: list[dict] | None = None,
    document: dict | None = None,
    artifacts: list[dict] | None = None,
    knowledge: dict | None = None,
    grounding: dict | None = None,
    clarification: dict | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
    degraded: list[str] | None = None,
    confidence_band: str | None = None,
    route: dict | None = None,
    trace: dict | None = None,
    input_modality: str | None = None,
    input: dict | None = None,
    content: str | None = None,
    knowledge_sources: list[str] | None = None,
    verified: bool | None = None,
) -> dict:
    """Assemble a `response.v1` envelope from a turn's computed fields → JSON dict.

    Every argument is optional; absent ones are omitted from the output. Never
    raises — a bad field degrades to omission, so it can't break the `done` event.
    """
    # §12: the input modality (text | image | document | audio | multimodal)
    # rides meta so every consumer can branch on it; the richer `input` block
    # (when provided) carries the per-kind counts.
    _modality = input_modality or (input or {}).get("modality")
    meta = {
        "model": model,
        "difficulty": difficulty,
        "latency_ms": latency_ms,
        "degraded": list(degraded or []),
        "confidence_band": confidence_band,
        "route": route,
        "modality": _modality,
        "trace_id": (trace or {}).get("id"),
        "trace": trace or None,
    }
    meta = {k: v for k, v in meta.items() if v not in (None, [], {})}
    # Response fingerprint (Phase 6 #22) — reproducibility/provenance id over the
    # answer (or prompt) + model + app version + sources. Additive + fail-open.
    # Only for substantive turns, so a bare/empty envelope keeps an empty meta.
    if model or content or resolved_prompt:
        try:
            from app.core.update_check import APP_VERSION
            from app.response_arch.fingerprint import response_fingerprint
            fp = response_fingerprint(
                content=content or resolved_prompt or "",
                model=model, app_version=APP_VERSION,
                knowledge_sources=knowledge_sources, verified=verified,
            )
            if fp:
                meta["fingerprint"] = fp
        except Exception:  # noqa: BLE001
            pass
    answer: dict = {"incomplete": bool(incomplete)}
    if answer_shape:
        answer["shape"] = answer_shape
    # Ensure the input block, when present, always carries its modality.
    _input = None
    if input:
        _input = dict(input)
        _input.setdefault("modality", _modality or "text")
    elif input_modality and input_modality != "text":
        _input = {"modality": input_modality}
    env = ResponseEnvelope(
        conversation_id=conversation_id,
        message_id=message_id,
        input=_input,
        intent=intent or None,
        difficulty=difficulty,
        resolved_prompt=resolved_prompt,
        topic_shift=bool(topic_shift),
        answer=answer,
        suggestions=list(suggestions or []),
        document=document or None,
        artifacts=list(artifacts or []),
        knowledge=knowledge or None,
        grounding=grounding or None,
        clarification=clarification or None,
        meta=meta,
    )
    return env.as_json()


def structure_suggestions(items, *, source: str = "profile", intent_of=None):
    """Normalize raw follow-up suggestions (plain strings or dicts) into envelope
    suggestion objects — ``{text, source, intent_hint?}`` (Architecture §6).

    `source` tags where the suggestion came from (profile | knowledge_graph |
    memory_graph). `intent_of` is an optional `str -> intent|None` callable used
    to tag each suggestion with the intent it would trigger (best-effort). Dicts
    that already carry `text`/`source`/`intent_hint` are passed through. Never
    raises; bad items are dropped.
    """
    out: list[dict] = []
    for it in items or []:
        try:
            if isinstance(it, dict) and str(it.get("text", "")).strip():
                s = {"text": str(it["text"]).strip(),
                     "source": it.get("source") or source}
                if it.get("intent_hint"):
                    s["intent_hint"] = it["intent_hint"]
                out.append(s)
            elif isinstance(it, str) and it.strip():
                s = {"text": it.strip(), "source": source}
                if intent_of is not None:
                    hint = intent_of(it)
                    if hint:
                        s["intent_hint"] = hint
                out.append(s)
        except Exception:  # noqa: BLE001 — skip a bad item, never fail the turn
            continue
    return out


_MODALITY_ORDER = ("image", "document", "audio")


def detect_input_modality(
    *,
    text: str | None = None,
    images=None,
    files=None,
    audio=None,
) -> str:
    """Classify a turn's input modality (Architecture §12).

    Returns ``text`` (no attachments), the single non-text kind (``image`` /
    ``document`` / ``audio``) when exactly one is present — text alongside is
    normal and doesn't change the primary modality — or ``multimodal`` when more
    than one non-text kind is present. Pure; never raises.
    """
    kinds = []
    if images:
        kinds.append("image")
    if files:
        kinds.append("document")
    if audio:
        kinds.append("audio")
    if not kinds:
        return "text"
    if len(kinds) == 1:
        return kinds[0]
    return "multimodal"


def build_input(
    *,
    text: str | None = None,
    images=None,
    files=None,
    audio=None,
) -> dict:
    """A compact, reload-stable `input` descriptor: the modality + per-kind
    counts/flags (never raw bytes). Pure; never raises."""
    return {
        "modality": detect_input_modality(
            text=text, images=images, files=files, audio=audio),
        "text": bool((text or "").strip()),
        "images": len(images or []),
        "files": len(files or []),
        "audio": bool(audio),
    }


def structure_artifacts(items, *, default_modality: str = "text"):
    """Normalize output artifacts into envelope objects (Architecture §12).

    Each artifact keeps its own fields but is guaranteed a ``kind`` and a
    ``modality`` so a client can render code / image / chart / diagram
    uniformly. Non-dict / empty items are dropped. Never raises.
    """
    out: list[dict] = []
    for it in items or []:
        try:
            if not isinstance(it, dict):
                continue
            a = dict(it)
            a.setdefault("kind", "code")
            a.setdefault("modality", _artifact_modality(a.get("kind"))
                         or default_modality)
            out.append(a)
        except Exception:  # noqa: BLE001 — skip a bad artifact, never fail the turn
            continue
    return out


def _artifact_modality(kind: str | None) -> str | None:
    k = (kind or "").lower()
    if k in ("image", "photo", "screenshot"):
        return "image"
    if k in ("chart", "plot", "diagram", "graph"):
        return "visual"
    if k in ("audio", "voice"):
        return "audio"
    if k in ("code", "text", "markdown", "document", "table"):
        return "text"
    return None


__all__ = [
    "ResponseEnvelope", "build_envelope", "structure_suggestions",
    "detect_input_modality", "build_input", "structure_artifacts",
]
