"""Schema-enforced structured output — the never-malformed invariant (§8.7).

One helper, ``structured()``, that call sites use instead of bespoke
``json.loads``-and-pray. It composes the existing dependency-free validator
(``app.llm.constrained``) with the routing engine into a repair→validate→
retry ladder:

    1. deterministic JSON-repair micro-pass (fenced blocks, trailing commas) +
       parse + validate against the schema;
    2. on failure, retry ONCE with the validation error fed back AND the model
       rotated (``avoid_model_db_id``) — different weights are the cheapest
       different answer;
    3. still failing → honest degrade: return the best-parsed object (or None)
       with ``degraded`` flags set, so the caller sees a typed result and the
       flywheel (§8.9) sees a ``schema_retry`` / ``schema_unvalidated`` signal.

The design leaves a seam for the true guaranteed-valid floor from §8.7 — a
grammar-constrained decode on a LOCAL model (``schema_to_gbnf`` below). That step
is intentionally NOT wired yet: this deployment has no local generation runtime
(``app.llm.local_infer`` is a planner, not an engine). The moment a local
llama.cpp/vLLM backend lands, the exhausted branch calls the grammar floor and
the invariant becomes a hard guarantee. Until then the ladder degrades honestly
rather than claiming a guarantee it can't keep.

Fail-open: any error in structuring returns a ``StructuredResult`` with the
error recorded — it never raises into the caller's business logic.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.llm import constrained

log = logging.getLogger(__name__)

# tier → the routing difficulty the structured call runs at.
_DIFF_BY_TIER = {
    "fast": "standard", "json": "standard", "coder": "standard",
    "reasoning": "hard", "expert": "expert",
}

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(s: str) -> str:
    """Remove ``, }`` / ``, ]`` — the single most common malformation from free
    models. Idempotent; leaves valid JSON untouched."""
    prev = None
    while prev != s:
        prev = s
        s = _TRAILING_COMMA.sub(r"\1", s)
    return s


def parse_with_repair(text: str, schema: dict) -> tuple[Any | None, list[str]]:
    """Parse ``text`` into a schema-validated object, applying the deterministic
    repair micro-pass. Returns ``(obj_or_None, errors)`` — ``obj`` is None only
    when nothing JSON-shaped could be recovered at all."""
    obj, errs = constrained.coerce(text, schema)
    if obj is not None:
        return obj, errs
    # coerce couldn't even parse — try the trailing-comma repair on the
    # extracted (de-fenced, outer-trimmed) payload.
    repaired = _strip_trailing_commas(constrained.extract_json(text or ""))
    obj2 = constrained.parse_json(repaired)
    if obj2 is not None:
        return obj2, constrained.validate(obj2, schema)
    return None, ["not valid JSON"]


@dataclass
class StructuredResult:
    obj: Any | None
    errors: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    model: str = ""

    @property
    def valid(self) -> bool:
        return self.obj is not None and not self.errors


def _schema_instruction(schema: dict, name: str) -> str:
    kind = "array" if schema.get("type") == "array" else "object"
    return (
        f"Respond with ONLY a single JSON {kind} named '{name}' that validates "
        "against this JSON Schema. No prose, no markdown fences, no comments.\n"
        "Schema:\n" + json.dumps(schema, sort_keys=True)
    )


def _record_schema_outcome(route, *, valid: bool, degraded: list) -> None:
    """§2.6 verify sink — a structured call IS the `extraction_json` profile, and
    we know the model, whether it validated, and whether it schema-retried. Feed
    that into the per-(identity, profile) scorecard so the router learns which
    models reliably return schema-valid JSON. Flag-gated + fail-open."""
    try:
        from app.core.config_loader import cfg
        if not bool(getattr(cfg.routing, "task_profiles", False)):
            return
        platform = getattr(route, "platform", None)
        model_id = getattr(route, "model_id", None)
        if not platform or not model_id:
            return
        from app.llm import scorecards
        from app.llm.identity import identity_key
        scorecards.record_verify_outcome(
            identity_key(platform, model_id), "extraction_json",
            passed=valid, schema_retried=("schema_retry" in (degraded or [])))
    except Exception:  # noqa: BLE001 — telemetry never breaks a structured call
        pass


async def structured(
    schema: dict,
    messages: list[dict],
    *,
    tier: str = "fast",
    name: str = "response",
    session_key: str | None = None,
    preferred_model_db_id: int | None = None,
    retries: int = 1,
    options: dict | None = None,
    augment_prompt: bool = True,
) -> StructuredResult:
    """Generate a schema-valid object. ``retries`` is the number of RE-tries
    after the first attempt (default 1 → up to 2 generations). Each retry feeds
    the validation error back and rotates the model."""
    from app.llm import engine  # lazy — avoid an import cycle at module load

    degraded: list[str] = []
    base_opts = dict(options or {})
    base_opts.setdefault("difficulty", _DIFF_BY_TIER.get(tier, "standard"))
    base_opts.setdefault("temperature", 0.0)
    base_opts["needs_json"] = True   # steer routing to a JSON-capable model

    msgs = list(messages)
    if augment_prompt:
        msgs = msgs + [{"role": "user", "content": _schema_instruction(schema, name)}]

    obj: Any | None = None
    errs: list[str] = ["no attempt"]
    avoid: int | None = None
    model = ""
    route = None                 # defined even if the first generation raises

    for attempt in range(max(1, retries + 1)):
        opts = dict(base_opts)
        if avoid is not None:
            opts["avoid_model_db_id"] = avoid
        try:
            text, route = await engine.route_and_complete(
                msgs, opts, session_key=session_key,
                preferred_model_db_id=(preferred_model_db_id
                                       if attempt == 0 else None),
            )
            model = (getattr(route, "display_name", "")
                     or getattr(route, "model_id", "") or model)
        except Exception as exc:  # noqa: BLE001 — NoRouteAvailable / provider err
            log.info("structured: generation failed (attempt %d): %s",
                     attempt, exc)
            errs = [f"generation failed: {exc}"]
            degraded.append("generation_error")
            break

        obj, errs = parse_with_repair(text, schema)
        if obj is not None and not errs:
            _record_schema_outcome(route, valid=True, degraded=degraded)
            return StructuredResult(obj, [], degraded, model)

        if attempt < retries:
            degraded.append("schema_retry")
            avoid = getattr(route, "model_db_id", None)
            msgs = msgs + [
                {"role": "assistant", "content": (text or "")[:2000]},
                {"role": "user", "content": (
                    "That was not valid against the schema. Errors:\n- "
                    + "\n- ".join(errs[:8])
                    + "\nReturn ONLY corrected JSON — no prose, no fences.")},
            ]

    # §8.7 rung 3 — the grammar floor. If the on-pod local runtime is enabled, a
    # final call to the local model with a GBNF grammar built from the schema is
    # GUARANTEED-valid JSON (llama.cpp cannot emit a non-matching token). Only
    # reached when the cloud attempts couldn't validate. Inert (skipped) when the
    # local runtime is off — then we degrade honestly below.
    try:
        from app.llm import router as _router
        local_id = await _router.local_model_db_id()
        if local_id is not None:
            gbnf = schema_to_gbnf(schema)
            floor_msgs = list(messages) + [
                {"role": "user", "content": _schema_instruction(schema, name)}]
            text, route = await engine.route_and_complete(
                floor_msgs,
                {"grammar": gbnf, "temperature": 0.0, "difficulty": "standard"},
                preferred_model_db_id=local_id)
            gobj, gerrs = parse_with_repair(text, schema)
            if gobj is not None and not gerrs:
                degraded.append("schema_local_floor")
                return StructuredResult(
                    gobj, [], degraded,
                    getattr(route, "display_name", "local"))
    except Exception:  # noqa: BLE001 — the floor must never raise into the caller
        pass

    # Exhausted and no local floor available — degrade honestly.
    _record_schema_outcome(route, valid=False, degraded=degraded)
    degraded.append("schema_unvalidated" if obj is None else "schema_invalid")
    return StructuredResult(obj, errs, degraded, model)


# ── The local grammar floor (ready for when a local engine lands) ────────────
_GBNF_PRIMITIVE = {
    "string": "string", "number": "number", "integer": "integer",
    "boolean": "boolean", "null": '"null"',
}


def schema_to_gbnf(schema: dict) -> str:
    """A GBNF grammar for the JSON subset ``constrained.validate`` covers, for
    the FUTURE local grammar-constrained floor (§8.7 rung 3). Not executed until
    a local generation backend exists — this exists so the floor is a drop-in.

    Covers: object (fixed properties, in schema order), array (items), enum of
    strings, and the JSON primitives. Best-effort; unknown shapes fall back to a
    permissive ``value``."""
    rules: list[str] = []
    counter = {"n": 0}

    def fresh(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    def emit(node: dict) -> str:
        if not isinstance(node, dict):
            return "value"
        if isinstance(node.get("enum"), list):
            opts = " | ".join(json.dumps(str(v)) for v in node["enum"])
            name = fresh("enum")
            rules.append(f"{name} ::= {opts}")
            return name
        t = node.get("type")
        if t == "object":
            props = node.get("properties", {}) or {}
            if not props:
                return "object"
            parts = []
            for k, sub in props.items():
                parts.append(f'"\\"{k}\\"" ws ":" ws {emit(sub)}')
            name = fresh("obj")
            body = ' ws "," ws '.join(parts)
            rules.append(f'{name} ::= "{{" ws {body} ws "}}"')
            return name
        if t == "array":
            item = emit(node.get("items", {}) or {})
            name = fresh("arr")
            rules.append(
                f'{name} ::= "[" ws ( {item} ( ws "," ws {item} )* )? ws "]"')
            return name
        return _GBNF_PRIMITIVE.get(t, "value")

    root = emit(schema)
    header = [
        f"root ::= ws {root} ws",
        'value ::= object | array | string | number | boolean | "null"',
        'object ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"',
        'array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"',
        r'string ::= "\"" ( [^"\\] | "\\" . )* "\""',
        'number ::= "-"? [0-9]+ ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?',
        'integer ::= "-"? [0-9]+',
        'boolean ::= "true" | "false"',
        'ws ::= [ \\t\\n]*',
    ]
    return "\n".join(header + rules)


__all__ = ["structured", "StructuredResult", "parse_with_repair",
           "schema_to_gbnf"]
