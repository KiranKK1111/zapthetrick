"""Constrained Decoding / Structured Outputs (roadmap Phase 6 #24).

Makes the Unified Response Object reliable: request schema-enforced output from
providers that support it (OpenAI-style ``response_format: json_schema``) and —
for EVERY provider — parse + validate the returned JSON against the schema, so a
malformed emission is caught (and repairable) instead of corrupting the envelope.

No new dependency: a small, dependency-free validator covering the JSON-Schema
subset we actually emit (type / required / properties / items / enum / bounds).
Fail-open and deterministic.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull the JSON payload out of a possibly-fenced / chatty response."""
    if not text:
        return ""
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    # Trim to the outermost object/array if there's leading/trailing prose.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if 0 <= i < j:
            return t[i:j + 1]
    return t


def parse_json(text: str) -> Any | None:
    try:
        return json.loads(extract_json(text))
    except Exception:  # noqa: BLE001
        return None


def _type_ok(value: Any, t: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(t, True)


def validate(obj: Any, schema: dict, *, path: str = "$") -> list[str]:
    """Return a list of human-readable errors ([] = valid) for the JSON-Schema
    subset we use."""
    errs: list[str] = []
    if not isinstance(schema, dict):
        return errs
    t = schema.get("type")
    if isinstance(t, str) and not _type_ok(obj, t):
        errs.append(f"{path}: expected {t}, got {type(obj).__name__}")
        return errs  # type wrong → deeper checks are noise
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object" or isinstance(obj, dict):
        props = schema.get("properties", {}) or {}
        for req in schema.get("required", []) or []:
            if not isinstance(obj, dict) or req not in obj:
                errs.append(f"{path}.{req}: required")
        if isinstance(obj, dict):
            for k, sub in props.items():
                if k in obj:
                    errs += validate(obj[k], sub, path=f"{path}.{k}")
    if (t == "array" or isinstance(obj, list)) and "items" in schema \
            and isinstance(obj, list):
        for idx, item in enumerate(obj):
            errs += validate(item, schema["items"], path=f"{path}[{idx}]")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} > maximum {schema['maximum']}")
    return errs


def coerce(text: str, schema: dict) -> tuple[Any | None, list[str]]:
    """Parse [text] and validate against [schema]. Returns (obj_or_None, errors).
    obj is None when it couldn't be parsed at all."""
    obj = parse_json(text)
    if obj is None:
        return None, ["not valid JSON"]
    return obj, validate(obj, schema)


def response_format(schema: dict, *, name: str = "response",
                    strict: bool = True) -> dict:
    """The OpenAI-style ``response_format`` payload for providers that support
    schema-enforced decoding. Callers gate on [supports_structured]."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


def supports_structured(model_meta: dict | None) -> bool:
    """Whether the model advertises JSON / response_format support."""
    if not model_meta:
        return False
    return bool(model_meta.get("supports_json")
               or model_meta.get("supports_response_format"))


__all__ = ["extract_json", "parse_json", "validate", "coerce",
           "response_format", "supports_structured"]
